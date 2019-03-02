# -*- coding: utf-8 -*-
# 主函数
# author = 'huangth'

import time

from feature_engineering import Processing
from model import model_main

if __name__ == "__main__":
    t0 = time.time()
    processing = Processing()
    processing.get_processing()
    print("Feature engineering has finished!")
    print("Cost {} s.".format(time.time() - t0))

    t1 = time.time()
    model_main()
    print("Model has trained!")
    print("Cost {} s.".format(time.time() - t1))

    print("ALL has finished!")
    print("Cost {} s.".format(time.time() - t0))