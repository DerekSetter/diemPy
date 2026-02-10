# API Reference

This is the API

```{eval-rst}
.. currentmodule:: diem
```

## diemtype

```{eval-rst}
.. automodule:: diem.diemtype
    :members: save_DiemType, load_DiemType, flip_polarity
    :exclude-members: DiemType

.. autoclass:: diem.diemtype.DiemType
   :members: sort,copy,polarize,apply_threshold,smooth,create_contig_matrix,get_intervals_of_state,intervals_to_bed
```


## contigs and intervals
```{eval-rst}
.. automodule:: diem.contigs
    :members: Interval, Contig, export_contigs_to_ind_bed_files
```

<!--
## polarization submodule
```{eval-rst}
.. automodule:: diem.polarize
    :members:
```
-->

<!--
## smoothing submodule
```{eval-rst}
.. automodule:: diem.smooth
    :members:
```
-->

<!--
## tests submodule
```{eval-rst}
.. automodule:: diem.tests
    :members:
```
-->

## plots submodule
```{eval-rst}
.. automodule:: diem.plots
    :members: GenomeSummaryPlot, GenomeMultiSummaryPlot,GenomicDeFinettiPlot, GenomicMultiDeFinettiPlot, GenomicContributionsPlot, diemPairsPlot, diemMultiPairsPlot, diemPlotPrepFromBedMeta, diemIrisFromPlotPrep, diemLongFromPlotPrep
```

This is the end of the API documentation
