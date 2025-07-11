
class PromptGen:
    def __init__(self):
        self.base_prompt = '''
            Instruction: You are a financial event analyst focused on analyzing the potential impacts of news reports on the market.
             Based on the given news content and current market structure, evaluate and output the affected companies (TICKER), 
             the type of impact (positive, negative, or neutral), and a score representing the strength of the impact (ranging from -10 to +10, 
             where -10 indicates a very negative impact, and +10 indicates a very positive impact). Provide specific company names and event descriptions for clarity and utility.

            Input Example: "Company A announces a partnership with Company B to jointly develop new technology, expected to significantly enhance production efficiency and increase market share."

            Output Format Example: { "impact_analysis": { "affected_companies": [ { "name": "Company A", "impact_type": "positive", "impact_score": 8 }, { "name": "Company B", "impact_type": "positive", "impact_score": 6 } ], "analysis": "The partnership between Company A and Company B is expected to enhance their technological capabilities and market competitiveness, likely increasing their revenues and stock prices." } }

            Input (you need to analyze): 
        '''

    def generate_instructions(self, news):
        return self.base_prompt + news + '''
            Provide your result, strictly following the output format in the example, without any additional output.
        '''

    def add_example(self, example):
        self.base_prompt += f'\n\nExample:\n{example}'

    def set_instruction(self, instruction):
        self.base_prompt = instruction