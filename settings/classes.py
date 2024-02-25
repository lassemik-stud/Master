class Dataset:
    def __init__(self, id, dataset, dataset_type, same_author, known_texts, unknown_texts):
        self.id = id
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.same_author = same_author
        self.known_texts = known_texts
        self.unknown_texts = unknown_texts
        self.unknown_paragraph_location = []

    def __repr__(self):
        return (f"TextData(id={self.id}, dataset={self.dataset}, dataset_type={self.dataset_type}, "
                f"same_author={self.same_author}, known_text={self.known_text}, "
                f"unknown_text={self.unknown_text})")
    
    def train(self):
        pass
    
    def test(self):
        pass