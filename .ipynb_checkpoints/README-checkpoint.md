# Splicing screen Efficacy Analysis with BAyesian StatisticS

## To run current version of seabass (hier_alt.py)
```python run_hier_alt.py ['cas13_LFCS_guides'] ['name_analysis'] ['output_directory']```

python /gpfs/commons/home/kisaev/seabass/src/run_hier_alt.py $file_name noR1 /gpfs/commons/groups/knowles_lab/Cas13Karin/analysis/seabass/

- this will generate an output directory with posterior values for guides, junctions and genes as well as summary plots 

## TODO
- multiple days
- deal with bimodality: are these just 0s? (use Harm's pipeline? Let Karin do this)
- learn prior on efficacy? Beta prior done. mixture?
- learn prior on essentiality? Variance done. 
- should prior on essentiality have negative mean? (in essential arm) 
- hierarchy over genes/junctions
- version 1: lfc = guide_efficacy * junction_targetability * gene_essentiality [done]
- version 2: junction_essentiality ~ N( gene_essentiality, sigma2 ) [done]
- allow conditioning efficacy (and essentiality?) on prediction from Andrew's model
- conditioning on isoform(junction) expression for isoform arm