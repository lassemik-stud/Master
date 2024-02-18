class Dataset:
    def __init__(self, id, dataset, dataset_type, same_author, known_text, unknown_text):
        self.id = id
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.same_author = same_author
        self.known_text = known_text
        self.unknown_text = unknown_text

    def __repr__(self):
        return (f"TextData(id={self.id}, dataset={self.dataset}, dataset_type={self.dataset_type}, "
                f"same_author={self.same_author}, known_text={self.known_text}, "
                f"unknown_text={self.unknown_text})")
    
    def train(self):
        pass
    
    def test(self):
        pass