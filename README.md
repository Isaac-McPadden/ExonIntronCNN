Under construction


Goal is to produce a neural network that can find intron and exon boundaries in a 5000 bp sequence (Current F1 PB of 0.6352 for data with sparseness of ~1.3/1000)

Model is only training on 5000 bp sections that have intron exon boundaries, so it will need an eventual feeder model. 

GTF and Fasta files can be found on this page: https://www.gencodegenes.org/human/

GTF file is the 'Basic gene annotation' with CHR as the 'regions' and can be direct downloaded here: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.basic.annotation.gtf.gz

I unzipped and saved it as 'basic_annotations.gtf'

The fasta file is 'Genome sequence (GRCh38.p14)' with ALL as the 'regions' and can be direct downloaded here: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/GRCh38.p14.genome.fa.gz

I unzipped and saved it as 'chr_genome.fa'


Repository Structure

There are 4 main folders: Datasets, Models, Notebooks, and Other Information

Datasets only contains FinalIntronExonDF.csv and nothing else on github because the datasets are big

Models has the Best Models folder which is the best models (by validation F1 score) I've managed to train so far. It also has the checkpoints subfolder which is used a lot by the code.  I move checkpoints into a Training Data subfolder.  For tuning trials, I have a Tuning Data subfolder. I have them in the repo but they are empty.

Notebooks is split into Curated Notebooks and Experimentation Notebooks. Curated Notebooks are streamlined step by step code cells with as much of the mess from testing stuff kept out.  Experimentation Notebooks is the Notebooks I was actually working in and using to create messes.

Other Information is other useful info for which I did not have a better spot.



To do:

Build out an experiment plan rather than continue with the throw-stuff-out-and-see-what-sticks method (F1 PB of 0.6352)
- currently have a general outline

Build infrastructure to make experiment streamlined and self documenting

With any luck produce a model that can get a validation score in the 0.90s

Produce pipeline version of code so anyone can use it, particularly for data transformation and feeding to model for eval

Build different model that can read 5000 bp sequence and classify it as having introns and/or exons 