
def get_reaction_dict(reaction):
    M = reaction.metabolites
    return {m.name : M[m] for m in M}

class MetaReaction:
    def __init__(self, _id):
        self.id = _id
        self.reactions = []
        self.reaction_dicts = []

    def __str__(self):
        return f"<{self.id} {self.is_consistent()} {self.reaction_dicts}>"
        
    def add_reaction(self, reaction):
        self.reactions.append(reaction)
        self.reaction_dicts.append(get_reaction_dict(reaction))

    def is_consistent(self) -> bool:
        return all([self.reaction_dicts[0] == r for r in self.reaction_dicts[1:]])