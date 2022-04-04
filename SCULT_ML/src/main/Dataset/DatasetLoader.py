from abc import ABC, abstractmethod
from typing import Iterable


class Dataset(ABC):
    @abstractmethod
    def makeDataset(self):
        pass


class LoadDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def makeDataset(self):
        return self.dataset.get_dataset()

    def retrieve_model(self, val):
        return self.dataset.retrieve_model(val)

    def set_age(self, dataset):
        self.dataset = dataset

    def load_instance(self):
        return self.dataset





class DatasetFactory:
    def __init__(self, factory):
        self.factory = factory

    def load_dataset(self):
        return self.factory.makeDataset()

    def retrieve_model(self, val):
        return self.factory.retrieve_model(val)

    def load_instance(self):
        return self.factory.load_instance()

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)

    def __ne__(self, o: object) -> bool:
        return super().__ne__(o)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)

    def __dir__(self) -> Iterable[str]:
        return super().__dir__()
