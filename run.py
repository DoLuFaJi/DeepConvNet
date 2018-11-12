import subprocess

print('30 16 0.01')
subprocess.call("python extractor.py -i 30 -b 4 -lr 0.01 -m 0.2 -n dodo2 -t 0", shell=True)
subprocess.call("python extractor.py -i 30 -b 4 -lr 0.01 -m 0.5 -n dodo2 -t 0", shell=True)
subprocess.call("python extractor.py -i 30 -b 4 -lr 0.01 -m 0.9 -n dodo2 -t 0", shell=True)

# print('30 1 0.01')
# subprocess.call("python extractor.py -i 30 -b 1 -lr 0.01 -m 0.2 -n matin -t 0", shell=True)
# subprocess.call("python extractor.py -i 30 -b 1 -lr 0.01 -m 0.5 -n matin -t 0", shell=True)
# subprocess.call("python extractor.py -i 30 -b 1 -lr 0.01 -m 0.9 -n matin -t 0", shell=True)

# print('30 16 0.001')
# subprocess.call("python extractor.py -i 30 -b 16 -lr 0.001 -m 0.2 -n aprem -t 0", shell=True)
# subprocess.call("python extractor.py -i 30 -b 16 -lr 0.001 -m 0.5 -n aprem -t 0", shell=True)
# subprocess.call("python extractor.py -i 30 -b 16 -lr 0.001 -m 0.9 -n aprem -t 0", shell=True)
#
# print('30 1 0.001') USELESS
# subprocess.call("python extractor.py -i 30 -b 1 -lr 0.001 -m 0.2 -n dodo -t 0", shell=True)
# subprocess.call("python extractor.py -i 30 -b 1 -lr 0.001 -m 0.5 -n dodo -t 0", shell=True)
# subprocess.call("python extractor.py -i 30 -b 1 -lr 0.001 -m 0.9 -n dodo -t 0", shell=True)
#
# print('T 30 16 0.01')
# subprocess.call("python extractor.py -i 30 -b 16 -lr 0.01 -m 0.2 -n dodo_tuto -t 1", shell=True)
# subprocess.call("python extractor.py -i 30 -b 16 -lr 0.01 -m 0.5 -n dodo_tuto -t 1", shell=True)
# subprocess.call("python extractor.py -i 30 -b 16 -lr 0.01 -m 0.9 -n dodo_tuto -t 1", shell=True)
# #
# print('T 30 1 0.01') USELESS
# subprocess.call("python extractor.py -i 30 -b 1 -lr 0.01 -m 0.2 -n dodo_tuto -t 1", shell=True)
# subprocess.call("python extractor.py -i 30 -b 1 -lr 0.01 -m 0.5 -n dodo_tuto -t 1", shell=True)
# subprocess.call("python extractor.py -i 30 -b 1 -lr 0.01 -m 0.9 -n dodo_tuto -t 1", shell=True)
#
# print('T 30 16 0.001')
# subprocess.call("python extractor.py -i 30 -b 16 -lr 0.001 -m 0.2 -n dodo_tuto -t 1", shell=True)
# subprocess.call("python extractor.py -i 30 -b 16 -lr 0.001 -m 0.5 -n dodo_tuto -t 1", shell=True)
# subprocess.call("python extractor.py -i 30 -b 16 -lr 0.001 -m 0.9 -n dodo_tuto -t 1", shell=True)
#
# print('T 30 1 0.001') USELESS
# subprocess.call("python extractor.py -i 30 -b 16 -lr 0.001 -m 0.2 -n dodo_tuto -t 1", shell=True)
# subprocess.call("python extractor.py -i 30 -b 16 -lr 0.001 -m 0.5 -n dodo_tuto -t 1", shell=True)
# subprocess.call("python extractor.py -i 30 -b 16 -lr 0.001 -m 0.9 -n dodo_tuto -t 1", shell=True)
