def r2_adjust(r2, sample_num, feature_num):
    return 1-(1-r2)*(sample_num-1)/(sample_num-feature_num-1)
