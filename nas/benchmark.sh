#!/bin/bash

CONFIG_STR='{"op_keys":["double_bnconv_3","identity","add"],"red_op_keys":["conv_1","double_bnconv_3","add"],"model_name":"test"}'
python grid-point.py --config-str $CONFIG_STR
