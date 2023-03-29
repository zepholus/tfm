from abc import ABC, abstractmethod
 
class ForecastingModel(ABC):
        
    @abstractmethod
    def fit(self, df):
        pass



    


