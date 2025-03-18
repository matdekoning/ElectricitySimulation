import json
import random
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="genetic_algorithm.log",
        filemode="a",  # Notice "append" rather than "w"
    )
    return logging.getLogger()

with open('genes.json', 'r') as json_file:
    gene_source = json.load(json_file)
with open('cost.json', 'r') as json_file:
    cost_source = json.load(json_file)


df = pd.read_csv('Production_and_Load.csv')[['datetime','Bulgaria','Italy','Munchen','Spain','VlakteRaan','Arkona','Load']]
# Nuclear?
# df['Munchen'] = 1

def run_individual(individual):
    """
    Helper function to run simulation in a separate process.
    Returns the same individual object after updating its black_out, etc.
    """
    individual.run_simulation()
    return individual

class Individual:
    def __init__(self, genes=None):
        self.length = len(gene_source.keys())
        gene_values = genes if genes is not None else [random.choice([0, 500, 1000,2000]) for _ in range(self.length)]
        self.genes = gene_values
        self.get_network()
        self.base_cost = self.calculate_base_costs()
        self.black_out_gwh = 0

    def __repr__(self):
        return f"Individual({self.network}, base_cost={self.base_cost})"

    def get_network(self):
        self.network = {}
        for key, val in zip(gene_source.keys(), self.genes):
            self.network[key] = val
    
    def calculate_base_costs(self):
        """
        Calculate the base costs of the individual.
        """
        summer = 0
        for location, gwh in self.network.items():
            if location in ['Munchen','Bulgaria','Italy','Spain']:
                summer += gwh*cost_source[f'{location}_solar']
            elif location in ['Arkona','VlakteRaan']:
                summer += gwh*cost_source[f'{location}_wind']
            else:
                summer += gwh*cost_source[location]
        return summer

    def run_simulation(self, get_df=False):
        # Create a working copy of the original dataframe
        df_sub = df.copy()
        locations = ['Bulgaria', 'Italy', 'Spain', 'Arkona', 'VlakteRaan']

        # Initialize columns based on network configuration
        for column in self.network.keys():
            if '-' not in column and '_' not in column:
                # Direct production capacity
                df_sub[column] = df_sub[column] * self.network[column]
            elif '-' in column:
                # Transmission capacity between nodes
                source = column.split('-')[0]
                df_sub[column] = self.network[column]
            else:
                # Battery capacity: initialize to 0, set initial row to network value
                df_sub[column] = 0
                df_sub.at[0, column] = self.network[column]

        # Set initial load at time 0 to 0 (or as desired)
        df_sub.at[0, 'Load'] = 0.0

        cols_to_convert = list(self.network.keys())
        df_sub[cols_to_convert] = df_sub[cols_to_convert].astype(float)

        df_sub['Balance'] = df_sub['Load']

        df_sub['System need'] = (df_sub['Load']/(
            df_sub['Munchen'] +
            df_sub[['Bulgaria', 'Bulgaria-Munchen']].min(axis=1) +
            df_sub[['Italy', 'Italy-Munchen']].min(axis=1) +
            df_sub[['Spain', 'Spain-Munchen']].min(axis=1) +
            df_sub[['Arkona', 'Arkona-Munchen']].min(axis=1) +
            df_sub[['VlakteRaan', 'VlakteRaan-Munchen']].min(axis=1)
        )).clip(upper=1)

        # Export production
        for location in locations:
            amount = (df_sub[[f'{location}', f'{location}-Munchen']].min(axis=1) * df_sub['System need'])
            df_sub['Balance'] = df_sub['Balance'] - amount
            df_sub[location] = df_sub[location] - amount
            df_sub[f'{location}-Munchen'] = df_sub[f'{location}-Munchen'] - amount
        df_sub['Balance'] = df_sub['Balance'] - (df_sub['Munchen'] * df_sub['System need'])
        df_sub['Munchen'] = df_sub['Munchen'] - (df_sub['Munchen'] * df_sub['System need'])


        # Batteries
        for i in range(1,len(df_sub.index)):
            balance = df_sub.loc[i, 'Balance']
            # decharge batteries
            if balance > 1:
                bat_munchen = df_sub.loc[i-1, 'Munchen_battery']
                bat_bulgaria = min(df_sub.loc[i-1, 'Bulgaria_battery'],df_sub.loc[i, 'Bulgaria-Munchen'])
                bat_italy = min(df_sub.loc[i-1, 'Italy_battery'],df_sub.loc[i, 'Italy-Munchen'])
                bat_spain = min(df_sub.loc[i-1, 'Spain_battery'],df_sub.loc[i, 'Spain-Munchen'])
                bat_vlakteraan = min(df_sub.loc[i-1, 'VlakteRaan_battery'],df_sub.loc[i, 'VlakteRaan-Munchen'])
                bat_arkona = min(df_sub.loc[i-1, 'Arkona_battery'],df_sub.loc[i, 'Arkona-Munchen'])
                battery_cap = sum([bat_munchen, bat_bulgaria, bat_italy, bat_spain, bat_vlakteraan,  bat_arkona])

                

                if battery_cap > balance:
                    factor = balance/battery_cap
                    df_sub.at[i, 'Balance'] = 0
                    df_sub.at[i, f'Munchen_battery'] = df_sub.loc[i-1, 'Munchen_battery']-(bat_munchen*factor)
                    df_sub.at[i, f'Bulgaria_battery'] = df_sub.loc[i-1, 'Bulgaria_battery']-(bat_bulgaria*factor)
                    df_sub.at[i, f'Italy_battery'] = df_sub.loc[i-1, 'Italy_battery']-(bat_italy*factor)
                    df_sub.at[i, f'Spain_battery'] = df_sub.loc[i-1, 'Spain_battery']-(bat_spain*factor)
                    df_sub.at[i, f'VlakteRaan_battery'] = df_sub.loc[i-1, 'VlakteRaan_battery']-(bat_vlakteraan*factor)
                    df_sub.at[i, f'Arkona_battery'] = df_sub.loc[i-1, 'Arkona_battery']-(bat_arkona*factor)
                else:
                    df_sub.at[i, f'Munchen_battery'] = df_sub.loc[i-1, 'Munchen_battery']
                    df_sub.at[i, f'Bulgaria_battery'] = df_sub.loc[i-1, 'Bulgaria_battery']
                    df_sub.at[i, f'Italy_battery'] = df_sub.loc[i-1, 'Italy_battery']
                    df_sub.at[i, f'Spain_battery'] = df_sub.loc[i-1, 'Spain_battery']
                    df_sub.at[i, f'VlakteRaan_battery'] = df_sub.loc[i-1, 'VlakteRaan_battery']
                    df_sub.at[i, f'Arkona_battery'] = df_sub.loc[i-1, 'Arkona_battery']          
            
            # charge batteries
            else:
                for location in locations:
                    potential = df_sub.loc[i-1, f'{location}_battery']+df_sub.loc[i, location]
                    max_battery = self.network[f'{location}_battery']
                    df_sub.at[i, f'{location}_battery'] = min(potential, max_battery)
                    # send to munchen
                    if potential > max_battery:
                        amount = min(df_sub.loc[i, f'{location}-Munchen'], potential-max_battery)
                        df_sub.at[i,'Munchen_battery'] = df_sub.loc[i,'Munchen_battery']+amount
                df_sub.at[i, f'Munchen_battery'] = min(df_sub.loc[i, 'Munchen_battery']+df_sub.loc[i-1, 'Munchen_battery']+df_sub.loc[i, 'Munchen'], self.network['Munchen_battery'])
        self.base_cost = int(round(self.base_cost+df_sub[df_sub.Balance>3].Balance.sum()*1000,0))
        self.black_out_gwh = df_sub[df_sub.Balance>3].Balance.sum()

        if get_df==True:
            return df_sub

class Population:
    def __init__(self, size=200):
        self.individuals = [Individual() for _ in range(size)]
        self.generation_number = 0
        self.logger = setup_logging()
        self.last_highscore = None

    def mutate(self, individual, mutation_rate=0.1):
        """Randomly mutate an individual's genes based on the mutation rate."""
        for i in range(len(individual.genes)):
            if random.random() < mutation_rate:
                mutation = random.choice([-100, -50, -10, 10, 50, 100])
                individual.genes[i] = individual.genes[i] + mutation
                individual.genes[i] = max(0, min(5000, individual.genes[i]))
        individual.base_cost = individual.calculate_base_costs()

    def crossover(self, parent1, parent2):
        """Perform single-point crossover for real-valued genes."""
        # choose the crossover position
        crossover_point = random.randint(1, len(parent1.genes) - 1)
        # slice and combine
        child_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        return Individual(genes=child_genes)
    
    def evaluate_fitness(self):
        """Evaluate the fitness of all individuals in the population."""
        with ProcessPoolExecutor() as executor:
            self.individuals = list(executor.map(run_individual, self.individuals))
        # for individual in self.individuals:
        #     individual.run_simulation()
        valid_individuals = [ind for ind in self.individuals]
        return sorted(valid_individuals, key=lambda ind: ind.base_cost)[:30]
    
  
    def next_generation(self, mutation_rate=0.1):
        self.logger.info(f"Starting generation {self.generation_number + 1}")
        # Retain the top 1 individual (elitism)
        top_individuals = self.evaluate_fitness()
        next_gen = [top_individuals[0]]  # Keep the best individual untouched
        self.logger.info(f"Top Individual {top_individuals[0].network}")
        self.logger.info(f"Total cost {top_individuals[0].base_cost}")
        if top_individuals[0].black_out_gwh > 1:
            self.logger.info(f"Black out GWh {top_individuals[0].black_out_gwh}")
        self.last_highscore = top_individuals[0].base_cost

        # Mutate the top 10 elites
        for elite in top_individuals:
            mutant = Individual(genes=elite.genes[:])
            self.mutate(mutant, mutation_rate)
            next_gen.append(mutant)

        # Generate the rest of the population through crossover
        while len(next_gen) < len(self.individuals):
            parent1, parent2 = random.sample(top_individuals, 2)  # Select parents from top individuals
            child = self.crossover(parent1, parent2)
            self.mutate(child, mutation_rate)  # Apply mutation to offspring
            next_gen.append(child)

        self.individuals = next_gen[:len(self.individuals)]  # Ensure population size consistency
        self.generation_number += 1