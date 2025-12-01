import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ciror/Desktop/robotica/tps/tpFinalA/install/tpFinalA'
