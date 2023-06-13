import code_bert_score
import pickle
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

class Evaluator:
    def __init__(self, problem_solutions, problem_condition, model, inp_lang, targ_lang):
        self.problem_solutions = problem_solutions
        self.problem_conditions = problem_condition
        self.model = model
        self.inp_lang = inp_lang
        self.targ_lang = targ_lang

    def generate_translations(self):
        predicted_solutions = []

        for test_problem in self.problem_conditions:
            solution = self.model.evaluate_sentence(self.inp_lang, self.targ_lang, test_problem)
            solution = self.targ_lang.sequences_to_texts(solution)
            predicted_solutions.append(solution)

        return predicted_solutions

    def bleu_scores(self):
        bleu_scores = []
        smoothie = SmoothingFunction().method4

        predicted_solutions = self.generate_translations()

        for i in range(len(self.problem_solutions)):
            problem = self.problem_solutions[i]
            predicted_solution = predicted_solutions[i]

            problem_tokens = problem.split()
            solution_tokens = predicted_solution[0].split()

            bleu_score = sentence_bleu([problem_tokens], solution_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu_score)

        return bleu_scores

    # def meteor_scores(self):
    #     meteor_scores = []

    #     predicted_solutions = self.generate_translations()

    #     for i in range(len(self.problem_solutions)):
    #         problem = self.problem_solutions[i]
    #         predicted_solution = predicted_solutions[i]
    #         print(f"Problem: {problem}\nSolution: {predicted_solution[0]}")
    #         score = meteor_score([problem], predicted_solution[0])
    #         meteor_scores.append(score)

    #     return meteor_scores
    
    def meteor_scores(self):
        meteor_scores = []
        
        predicted_solutions = self.generate_translations()
        
        for i in range(len(self.problem_solutions)):
            problem = self.problem_solutions[i]
            predicted_solution = predicted_solutions[i]
            
            problem_tokens = word_tokenize(problem)
            solution_tokens = word_tokenize(predicted_solution[0])
            
            score = meteor_score([problem_tokens], solution_tokens)
            meteor_scores.append(score)
        
        return meteor_scores


    def code_bert_scores(self, lang='python'):
        idf_dict_path = r'./idf_dicts/python_idf.pkl'

        predicted_solutions = self.generate_translations()
        predicted_solutions = [' '.join(solution) for solution in predicted_solutions]
        print(f"""For the problems: {self.problem_conditions}\n We got the predicted solutions: {predicted_solutions}""")

        with open(idf_dict_path, 'rb') as f:
            idf_dict = pickle.load(f)

        pred_results = code_bert_score.score(cands=predicted_solutions, refs=self.problem_solutions, no_punc=True, lang=lang, idf_dict=idf_dict)

        return pred_results  # returns 4 separate tuples for precision, recall, F1, F3 for each prediction-reference_solution pair
