from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

class Evaluator:
    def __init__(self, problems, test_set, model, inp_lang, targ_lang):
        self.problems = problems
        self.test_set = test_set
        self.model = model
        self.inp_lang = inp_lang
        self.targ_lang = targ_lang

    def generate_translations(self):
        predicted_solutions = []

        for test_problem in self.test_set:
            solution = self.model.evaluate_sentence(self.inp_lang, self.targ_lang, test_problem)
            solution = self.targ_lang.sequences_to_texts(solution)
            predicted_solutions.append(solution)
        
        return predicted_solutions
        
    def bleu_scores(self):
        bleu_scores = []
        smoothie = SmoothingFunction().method4  

        predicted_solutions = self.generate_translations()

        for i in range(len(self.problems)):
            problem = self.problems[i]
            predicted_solution = predicted_solutions[i]

            problem_tokens = problem.split()
            solution_tokens = predicted_solution[0].split()

            bleu_score = sentence_bleu([problem_tokens], solution_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu_score)

        return bleu_scores

