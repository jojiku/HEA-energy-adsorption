# Instructions

Idea - to predict EA from H/S adsorbates with Hight entropy alloy (['Ag', 'Co', 'Cr', 'Cu', 'Fe', 'H', 'Mn', 'Ni', 'S', 'Zn', 'Zr'])

1. Put dft files into root/train_data_dft folder

2. Create graph file from dft 
```
python train_lgnn/dft2graphs_site_specific.py
```

3. Train
```
python train_site_simple.py
```
