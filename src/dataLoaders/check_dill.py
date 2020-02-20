import dill
from tqdm import tqdm

BAND ='middle'

path_to_oldData = '/dat/'

print("Loading dill files...")
oldFile = open(path_to_oldData + BAND + '_2_data_3.dill', 'rb')
newFile = open('dat/' + BAND + '_2_data.dill', 'rb')
dat = dill.load(oldFile)
newDat = dill.load(newFile)
print("Data loaded.")

if len(dat) != len(newDat):
	raise ValueError("The old list has length " + str(len(dat)) + " , but the new list has length " + str(len(newDat)))

for i in tqdm(range(len(dat))):
	for key, value in dat[i].items():
		if key == 'band' or 'audio':
			continue
		elif key not in newDat[i].keys():
			print(dat[i].keys())
			print(newDat[i].keys())
			raise ValueError("Keys are different at list index " + str(i))
		elif value not in newDat[i].values():
			print(dat[i].values())
			print(newDat[i].values())
			raise ValueError("Found different values at list index " + str(i))

print("The dill files are the same!")