from copy import deepcopy

# Reduced Protein letters(7 letters)
def get_reduced_protein_letter_dict():
    rpdict = {}
    reduced_letters = [["A","G","V"],
                       ["I","L","F","P"],
                       ["Y","M","T","S"],
                       ["H","N","Q","W"],
                       ["R","K"],
                       ["D","E"],
                       ["C"]]
    changed_letter = ["A","B","C","D","E","F","G"]
    for class_idx, class_letters in enumerate(reduced_letters):
        for letter in class_letters:
            rpdict[letter] = changed_letter[class_idx]
    
    return rpdict


# Improved CTF 
class improvedCTF:
    def __init__(self, letters, length):
        self.letters = letters
        self.length = length
        self.dict = {}
        self.generate_feature_dict()
        
    def generate_feature_dict(self):
        def generate(cur_key, depth):
            if depth == self.length:
                return
            for k in self.letters:
                next_key = cur_key + k
                self.dict[next_key] = 0
                generate(next_key, depth+1)
                
        generate(cur_key="",depth=0)
        
        print("iterate letters : {}".format(self.letters))
        print("number of keys  : {}".format(len(self.dict.keys())))
        
    
    def get_feature_dict(self):
        for k in self.dict.keys():
            self.dict[k] = 0
            
        return deepcopy(self.dict)