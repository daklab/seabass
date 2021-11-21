# Splicing screen Efficacy Analysis with BAyesian StatisticS

TODO
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