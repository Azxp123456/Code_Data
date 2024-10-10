# MMSNet - a novel multimodal deep learning method
## Dependencies

The code was developed and tested using python 3.8.  
To install python dependencies run: pip install -r requirements.txt  

## Scripts

* PDB_to_DSSP.py - This script is used to convert PDB format files to DSSP format files containing secondary structure information on the https://www3.cmbi.umcn.nl/xssp/ website. 
* get_StructureData.py - This script is mainly used to obtain protein distance maps and extract secondary structure information.  
* Contact_Distance_Map.py - This script is used to process distance maps into contact maps as pre-processed data to be input.

## Data

* Grain protein sequence source data is available at https://www.uniprot.org/  
* The structural data can be downloaded at https://alphafold.com/download  
* Gene ontology term files in OBO format can be downloaded at https://geneontology.org/docs/download-ontology/