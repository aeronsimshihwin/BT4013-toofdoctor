from .data_loading import *
from .futuresInfo import *
from .future_stats import *
from .post_processing import *
from .pre_processing import *
from .technical_indicators import *
from .metrics import *
from .generateWindows import *

# Connect technical indicators to main.py
class Preprocessor:
    @staticmethod
    def wrap(func, num_inputs, num_outputs):
        return Preprocessor(num_inputs, num_outputs)(func)

    def __init__(self, num_inputs, num_outputs):
        """To use class instance as a decorator"""
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def __call__(self, func):
        """To use class instance as a decorator"""
        @functools.wraps(func)
        def helper(data: pd.DataFrame, input: str, output: str, *args, **kwargs):
            # Handle multi input
            if self.num_inputs == 1:
                processed_input = [data[input]]
            else:
                processed_input = [data[col] for col in input]
            result = func(*processed_input, *args, **kwargs)
            
            # Handle multi output
            if self.num_outputs == 1:
                processed_output = result
                data[output] = processed_output
            else:
                processed_output = np.vstack(result).T
                for i in range(len(output)):
                    data[output[i]] = processed_output[:, i]

            return data
        return helper
    
SMA = Preprocessor.wrap(SMA, num_inputs=1, num_outputs=1)
EMA = Preprocessor.wrap(EMA, num_inputs=1, num_outputs=1)
MACD = Preprocessor.wrap(MACD, num_inputs=1, num_outputs=1)
RSI = Preprocessor.wrap(RSI, num_inputs=1, num_outputs=1)
ATR = Preprocessor.wrap(ATR, num_inputs=3, num_outputs=1)
VPT = Preprocessor.wrap(VPT, num_inputs=2, num_outputs=1)
BBands = Preprocessor.wrap(BBands, num_inputs=1, num_outputs=2)
CCI = Preprocessor.wrap(CCI, num_inputs=3, num_outputs=1)