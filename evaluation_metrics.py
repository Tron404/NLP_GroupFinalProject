import code_bert_score
import pickle
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

class Evaluator:
    def __init__(self, problem_condition, problem_solutions, model, inp_lang, targ_lang):
        self.problem_conditions = problem_condition   # List of problem conditions
        self.problem_solutions = problem_solutions   # List of correct solutions to the problems
        self.model = model                           # Model to be used for translation
        self.inp_lang = inp_lang                     # Source language tokenizer
        self.targ_lang = targ_lang                   # Target language tokenizer

    def generate_translations(self):
        predicted_solutions = []  # Empty list to store the model's solutions

        # Iterate over all the problem conditions
        for test_problem in self.problem_conditions:
            # Generate model's solution for the current problem condition
            solution = self.model.evaluate_sentence(self.inp_lang, self.targ_lang, test_problem)
            solution = self.targ_lang.sequences_to_texts(solution)
            # Add the model's solution to the list
            predicted_solutions.append(solution)

        # Return the list of model's solutions
        return predicted_solutions

    # Method to calculate BLEU scores for the model's solutions
    def bleu_scores(self):
        bleu_scores = []  # Empty list to store the BLEU scores
        smoothie = SmoothingFunction().method4  # Smoothing function for calculating BLEU score

        predicted_solutions = self.generate_translations()  # Generate model's solutions

        # Iterate over all the correct solutions
        for i in range(len(self.problem_solutions)):
            problem = self.problem_solutions[i]
            predicted_solution = predicted_solutions[i]

            # Tokenize the correct solution and the model's solution
            problem_tokens = problem.split()
            solution_tokens = predicted_solution[0].split()

            # Calculate BLEU score for the current correct solution and model's solution
            bleu_score = sentence_bleu([problem_tokens], solution_tokens, smoothing_function=smoothie)
            # Add the BLEU score to the list
            bleu_scores.append(bleu_score)

        return bleu_scores


    # Method to calculate METEOR scores for the model's solutions
    def meteor_scores(self):
        meteor_scores = []  # Empty list to store the METEOR scores
        
        predicted_solutions = self.generate_translations()  # Generate model's solutions
        
        # Iterate over all the correct solutions
        for i in range(len(self.problem_solutions)):
            problem = self.problem_solutions[i]
            predicted_solution = predicted_solutions[i]
            
            # Tokenize the correct solution and the model's solution
            problem_tokens = word_tokenize(problem)
            solution_tokens = word_tokenize(predicted_solution[0])
            
            # Calculate METEOR score for the current correct solution and model's solution
            score = meteor_score([problem_tokens], solution_tokens)
            # Add the METEOR score to the list
            meteor_scores.append(score)
        
        # Return the list of METEOR scores
        return meteor_scores


    # Method to calculate CodeBERT scores for the model's solutions
    def code_bert_scores(self, lang='python'):
        idf_dict_path = r'./idf_dicts/python_idf.pkl'  # Path to the idf dictionary file

        predicted_solutions = self.generate_translations()  # Generate model's solutions
        predicted_solutions = [' '.join(solution) for solution in predicted_solutions]
        
        print(f"For the problems: {self.problem_conditions}\n We got the predicted solutions: {predicted_solutions}")

        # Open a pre-calculated idf dictionary file
        with open(idf_dict_path, 'rb') as f:
            idf_dict = pickle.load(f)  # Load the idf dictionary

            # Calculate CodeBERT scores for the correct solutions and model's solutions
            pred_results = code_bert_score.score(cands=predicted_solutions, refs=self.problem_solutions, no_punc=True, lang=lang, idf=idf_dict)

            return pred_results

