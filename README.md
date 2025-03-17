Under construction


Goal is to produce a neural network that can find intron and exon boundaries in a 5000 bp sequence (Current F1 PB of 0.6352 for data with sparseness of ~1.3/1000)

Model is only training on 5000 bp sections that have intron exon boundaries, so it will need an eventual feeder model. 

GTF and Fasta files can be found on this page: https://www.gencodegenes.org/human/

GTF file is the 'Basic gene annotation' with CHR as the 'regions' and can be direct downloaded here: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.basic.annotation.gtf.gz

I unzipped and saved it as 'basic_annotations.gtf'

The fasta file is 'Genome sequence (GRCh38.p14)' with ALL as the 'regions' and can be direct downloaded here: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/GRCh38.p14.genome.fa.gz

I unzipped and saved it as 'chr_genome.fa'



Currently restructuring the project because the scope has expanded as I've working on it

To do:

Fix pathnames in notebooks that are not going to work since I've moved things around

Get a curated notebook for the model thusfar

Update README so it actually describes the project repo

Build out an experiment plan rather than continue with the throw-stuff-out-and-see-what-sticks method (F1 PB of 0.6352)

Build infrastructure to make experiment streamlined and self documenting

With any luck produce a model that can get a validation score in the 0.90s

Produce pipeline version of code so anyone can use it, particularly for data transformation and feeding to model for eval

Build different model that can read 5000 bp sequence and classify it as having introns and/or exons 