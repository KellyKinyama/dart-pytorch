# **Vectors Nuts and Bolts: Documentation**

Here is the adventure of a vector from raw floats to a ranked search
result — documented step by step, tracing the code that lives under
[../../lib/core/vector_store](../../lib/core/vector_store).

The goal of this series is the same as FAISS' own tutorial: start
from a brute-force flat index, then earn every approximation
(cell-probe IVF, product quantization, HNSW graphs, refine wrappers,
pre-transforms) by seeing why the previous step was insufficient.
Along the way we cover persistence (the on-disk FAISS binary format
plus this port's `IxDT` tuning wrapper), auto-tuning, and the CLI
tools shipped in [../../bin](../../bin).

[0. INFRASTRUCTURE](./00-INFRASTRUCTURE.md)
<br>
[1. RUNNING THE FAISS DEMO](./01-RUNNING-THE-FAISS-DEMO.md)
<br>
[2. VECTORS AND THE INDEX CONTRACT](./02-VECTORS-AND-THE-INDEX-CONTRACT.md)
<br>
[3. INDEXFLAT: THE GROUND TRUTH](./03-INDEXFLAT-THE-GROUND-TRUTH.md)
<br>
[4. INDEXIVFFLAT: CELL-PROBE SEARCH](./04-INDEXIVFFLAT-CELL-PROBE-SEARCH.md)
<br>
[5. INDEXIVFPQ: PRODUCT QUANTIZATION](./05-INDEXIVFPQ-PRODUCT-QUANTIZATION.md)
<br>
[6. INDEXHNSW: GRAPH-BASED SEARCH](./06-INDEXHNSW-GRAPH-BASED-SEARCH.md)
<br>
[7. TRANSFORMS AND COMPOSED INDEXES](./07-TRANSFORMS-AND-COMPOSED-INDEXES.md)
<br>
[8. PERSISTENCE: FAISS FORMAT AND THE IXDT WRAPPER](./08-PERSISTENCE-FAISS-FORMAT-AND-IXDT.md)
<br>
[9. AUTO-TUNING, BENCHMARKING, AND THE CLI TOOLS](./09-AUTO-TUNING-AND-CLI-TOOLS.md)
<br>
[10. CONCLUSION](./10-CONCLUSION.md)
<br>
[11. INTEGRATING VECTOR INDEXES WITH LANGUAGE MODELS](./11-LANGUAGE-MODEL-INTEGRATION.md)
<br>
[12. RAG NUTS AND BOLTS: TRACING A SINGLE QUERY END-TO-END](./12-RAG-NUTS-AND-BOLTS.md)<br>

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Home: README](../../README.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: INFRASTRUCTURE&nbsp;&nbsp;&gt;](./00-INFRASTRUCTURE.md)

</div>
