* Package used to perform DCA: https://github.com/ddsjoberg/dcurves
* Docs: http://www.danieldsjoberg.com/dcurves/index.html
* MSKCC Tutorial: https://www.mskcc.org/sites/default/files/node/4509/documents/dca-tutorial-2015-2-26.pdf

### 1. Run DCA on sample data from the MSKCC tutorial:
```r
# Load data
setwd("C:/Users/Jan/Downloads/decisioncurveanalysis")
df = read.delim("dca.txt", header=TRUE, sep="\t")

# Load dcurves library
library(dcurves)

# Run DCA
dca(cancer ~ cancerpredmarker + famhistory, 
    data = df_binary,
    thresholds = seq(0, 0.35, by = 0.01),
    label = list(
        cancerpredmarker = "Prediction Model",
        famhistory = "Family History")
    )
```


### Our data
```r
# Net Benefit
dca(labels ~ preds, data=birads4, thresholds=seq(0, 0.35, by=0.01)) %>% plot(smooth=TRUE)

# Standardized Net Benefit
dca(labels ~ preds, data=birads4, thresholds=seq(0, 0.35, by=0.01)) %>% standardized_net_benefit() %>% plot(smooth=TRUE)

# Net Interventions Avoided
dca(labels ~ preds, data=birads4, thresholds=seq(0, 0.35, by=0.01)) %>% net_intervention_avoided(nper=1000) %>% plot(smooth=TRUE)

# Plotting Net Interventions Avoided with ggplot2
# -- 1: create object
dcaobj <- dca(labels ~ preds, data=birads4, thresholds=seq(0, 0.30, by=0.005)) %>% net_intervention_avoided(nper=1000)
# -- 2: print ggplot2 command
dca(labels ~ preds, data=birads4, thresholds=seq(0, 0.30, by=0.005)) %>% net_intervention_avoided() %>% plot(show_ggplot_code=TRUE, smooth=TRUE)
# -- 3: use the printed command to display saved object
as_tibble(dcaobj) %>%
  dplyr::filter(!is.na(net_intervention_avoided), !(variable %in% 
    c("all", "none"))) %>%
  ggplot(aes(x = threshold, y = net_intervention_avoided, color = label)) +
  stat_smooth(method = "loess", se = FALSE, formula = "y ~ x", 
    span = 0.2, size=2) +
  coord_cartesian(ylim = c(0, 1000)) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_y_continuous(trans='pseudo_log') + 
  labs(x = "Threshold Probability", y = "Net reduction in interventions\nper 1000 patients", 
    color = "") +
  theme_light() + 
  theme(axis.text.x=element_text(size=12), axis.text.y=element_text(size=12), axis.title.x=element_text(size=12), axis.title.y=element_text(size=13), legend.position="None") +
  theme(plot.background = element_blank(), panel.border = element_blank(), panel.background = element_blank()) +
  theme(axis.line = element_line(color = 'black'))


# Nicely Formatted Net Interventions Avoided with Log Scale
as_tibble(dcaobj) %>%
    dplyr::filter(!is.na(net_intervention_avoided), !(variable %in% 
                                                        c("all", "none"))) %>%
    ggplot(aes(x = threshold, y = net_intervention_avoided, color = label)) +
    geom_line(size=1) +
    coord_cartesian(ylim = c(0, 250), xlim=c(0,0.1)) +
    scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
    scale_y_continuous(trans="pseudo_log", breaks=scales::trans_breaks('log10', function(x) 5^x, n=9)) + 
    labs(x = "Threshold Probability", y = "Net reduction in interventions\nper 1000 patients", 
        color = "") +
    theme_light() + 
    theme(axis.text.x=element_text(size=12), axis.text.y=element_text(size=12), axis.title.x=element_text(size=12), axis.title.y=element_text(size=13), legend.position="None") +
    theme(plot.background = element_blank(), panel.border = element_blank(), panel.background = element_blank()) +
    theme(axis.line = element_line(color = 'black'))

```

With the rmda package:
```r
library(rmda)

# save plot as 4.5in x 6in landscape
decision_curve(labels ~ preds, data=birads4, bootstraps=2000, thresholds=seq(0, 0.3, by=0.005), fitted.risk=TRUE, policy='opt-out') %>% plot_decision_curve(standardize = T, legend.position = "none", curve.names = c("AI System"))
```

