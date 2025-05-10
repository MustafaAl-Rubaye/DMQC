import numpy as np


def calculate_weights(labels_gpu):

    labels = np.array(labels_gpu.to("cpu"))
    wb_pix = []
    b_pix = []
    m_pix = []
    sf_pix = []
    n_pix = []
    for i in labels:
        wb = i[0]
        b = i[1]
        m = i[2]
        sf = i[3]
        n = i[4]

        wb_pix.append(wb.sum())
        b_pix.append(b.sum())
        m_pix.append(m.sum())
        sf_pix.append(sf.sum())
        n_pix.append(n.sum())

    wb_sum = sum(i for i in wb_pix)
    b_sum = sum(i for i in b_pix)
    m_sum = sum(i for i in m_pix)
    sf_sum = sum(i for i in sf_pix)
    n_sum = sum(i for i in n_pix)

    all = wb_sum + b_sum + m_sum + sf_sum + n_sum

    wb_w = all / wb_sum
    b_w = all / b_sum
    m_w = all / m_sum
    sf_w = all / sf_sum
    n_w = all / n_sum

    return [wb_w, b_w, m_w, sf_w, n_w]
