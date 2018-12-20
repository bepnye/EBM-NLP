import glob

fnames = glob.glob('base_annotations/annotations/*/train/*.ann') + \
         glob.glob('base_annotations/annotations/*/test/*/*.ann')

for f in fnames:
  labels = open(f).read().split(',')
  bio = ['O' if labels[0] == '0' else 'B']
  for l in labels[1:]:
    if l == '0': bio.append('O')
    else:
      if bio[-1] == 'O': bio.append('B')
      else: bio.append('I')
  with open(f.replace('.ann', '.bio'), 'w') as fp:
    fp.write(','.join(bio))

