'''
queryname = 'ALZ'
query = {'CR1', 'TREM2', 'ABCA7', 'EPHA1', 'CD2AP', 'CLU', 'BIN1', 'SORL1', 'UNC5C', 'CD33'}
expansion = set()
edgespernodemax = 5 # 10
r2min = 0 # 0.5**2
edges = set()
genes = set()
with open('average_pearson_similarity_top10.txt', 'rt') as fr:
    for line in fr:
        entries = [x.strip() for x in line.split('\t')]
        geneA = entries[0]
        if geneA in query:
            for other in entries[3:3+edgespernodemax]:
                geneB, weight = other.split(',')
                if float(weight)**2 >= r2min:
                    expansion.add(geneB)
query = query.union(expansion)
with open('average_pearson_similarity_top10.txt', 'rt') as fr, open('average_pearson_similarity_edgespernodemax{0!s}_r2min{1!s}_{2}top10_edges.txt'.format(edgespernodemax, r2min, queryname), 'wt') as fw:
    for line in fr:
        entries = [x.strip() for x in line.split('\t')]
        geneA = entries[0]
        for other in entries[3:3+edgespernodemax]:
            geneB, weight = other.split(',')
            if float(weight)**2 >= r2min and (geneA in query or geneB in query) and (geneB, geneA, weight) not in edges and (geneA, geneB, weight) not in edges:
                edges.add((geneA, geneB, weight))
                genes.add(geneA)
                genes.add(geneB)
                fw.write('\t'.join([geneA, geneB, weight]) + '\n')
gene_count = {}
genes = list(genes)
for geneA, geneB, weight in edges:
    if geneA not in gene_count:
        gene_count[geneA] = 1
    else:
        gene_count[geneA] += 1
    if geneB not in gene_count:
        gene_count[geneB] = 1
    else:
        gene_count[geneB] += 1
singlegenes = set()
for gene, count in gene_count.items():
    if count == 1:
        singlegenes.add(gene)
with open('average_pearson_similarity_edgespernodemax{0!s}_r2min{1!s}_{2}top10_edges_noterminalnodes.txt'.format(edgespernodemax, r2min, queryname), 'wt') as fw:
    for geneA, geneB, weight in edges:
        if geneA not in singlegenes and geneB not in singlegenes:
            fw.write('\t'.join([geneA, geneB, weight]) + '\n')
'''

# queryname = 'ALZ'
# query = {'CR1', 'TREM2', 'ABCA7', 'EPHA1', 'CD2AP', 'CLU', 'BIN1', 'SORL1', 'UNC5C', 'CD33'}
#queryname = 'ISG15'
#query = {'ISG15'}
#queryname = 'TNT7'
#query = {'PAK1', 'WASL', 'PTK2', 'TEK', 'GAP43', 'TTYH1', 'GJA1'}
queryname = 'TNTsig'
query = {'GAP43', 'VGF', 'TTYH1', 'KCNF1', 'OLIG2', 'CA9', 'OLIG1', 'H19', 'FXYD6', 'HES5', 'NDUFA4L2', 'FAM69C', 'TEK', 'C1orf61', 'C1orf115', 'GRIK1', 'HEPACAM', 'FAM181B', 'CDC20', 'ASCL1', 'LUM', 'ID3', 'CFI', 'SLC47A2', 'FRZB', 'INHBE', 'GLIPR1', 'VIPR1', 'IFITM1', 'IFI27', 'RGS4', 'SLPI', 'CCL2', 'GPNMB', 'AKR1C3', 'ALDH3A1', 'PCP4', 'CXCL14', 'MON1B'}
expansion = set()
edgespernodemax = 89 # 18 = 99.9%, 89 = 99.5%
r2min = 0.99995020489607811**2
edges = set()
with open('autoencoder_similarity_clr_normalized_top100.txt', 'rt') as fr:
    for line in fr:
        entries = [x.strip() for x in line.split('\t')]
        geneA = entries[0]
        if geneA in query:
            for other in entries[3:3+edgespernodemax]:
                geneB, weight = other.split(',')
                if float(weight)**2 >= r2min:
                    expansion.add(geneB)
                    edges.add((geneA, geneB, weight))
specificexpansion = set()
with open('autoencoder_similarity_clr_normalized_top100.txt', 'rt') as fr:
    for line in fr:
        entries = [x.strip() for x in line.split('\t')]
        geneA = entries[0]
        if geneA in expansion:
            for other in entries[3:3+edgespernodemax]:
                geneB, weight = other.split(',')
                if float(weight)**2 >= r2min and geneB in query:
                    specificexpansion.add(geneA)
                    break
exp_qry = {}
for geneA, geneB, weight in edges:
    if geneB in specificexpansion:
        if geneB not in exp_qry:
            exp_qry[geneB] = [','.join([geneA, weight])]
        else:
            exp_qry[geneB].append(','.join([geneA, weight]))
with open('exp_qry_clr_normalized_specific_edgesperqrymax{0!s}_r2min{1!s}_{2}.txt'.format(edgespernodemax, r2min, queryname), 'wt') as fw:
    for exp, qry in exp_qry.items():
        fw.write('\t'.join([exp, str(len(qry))] + qry) + '\n')
