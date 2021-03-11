import sys

#--- Settings ---
IGNORE_null = False
# AUTO_PAD_null = True # translation side
#--- Settings ---

def judge(pin, pout):
    acc = 0
    total = 0
    acc_with_null = 0
    total_with_null = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    df_line = 0
    t_line = 0

    df_list = []

    with open(pin, mode='r', encoding='utf-8') as fin:
        with open(pout, mode='r', encoding='utf-8') as fout:
            for tls, tgt in zip(fin, fout):
                tls = tls.split()
                tgt = tgt.split()
                len_tls = len(tls)
                len_tgt = len(tgt)
                if len_tls != len_tgt:
                    df_line += 1
                    df_list += [len_tls-len_tgt]
                    # if abs(len_tls - len_tgt) > 5:
                    #     print('--- ---')
                    #     print(' '.join(tgt))
                    #     print()
                    #     print(' '.join(tls))
                if len_tls >= len_tgt:
                    tls = tls[:len_tgt]
                else:
                    tls += ['<null>' for _ in range(len_tgt - len_tls)]
                for i in range(len_tgt):
                    if tgt[i] == '<null>':
                        if tls[i] == tgt[i]:
                            acc_with_null += 1
                        total_with_null += 1
                    else:
                        if tls[i] == tgt[i]:
                            acc_with_null += 1
                            acc += 1
                        total_with_null += 1
                        total += 1
                    # F1 score compute
                    if tgt[i] == '<null>':
                        if tls[i] == '<null>':
                            tn += 1 # 将负类预测为负类数
                        else:
                            fp += 1 # 
                    else:
                        if tgt[i] == tls[i]:
                            tp += 1
                        else:
                            fn += 1
                t_line += 1
    print('Accuracy : {}'.format(acc / total))
    print('Accuracy no null & ALL: {}'.format(acc / total_with_null))
    print('Accuracy with null: {}'.format(acc_with_null / total_with_null))
    print('different lens : ', df_line, ' total lines : ', t_line)
    print('F1 score:', 2 * tp /(2 * tp + fp + fn))
    print('F1 accuracy:', (tp + tn) / (tp + tn + fn + fp))
    print('F1 p:', tp / (tp + fp))
    print('F1 r:', tp / (tp + fn))
    print('tp=', tp, ';tn=',tn, ';fn=',fn,';fp=',fp)
    print('tp:非null标签预测正确的个数')
    print('tn:  null标签预测null标签的个数')
    print('fn:非null标签预测非null标签的个数')
    print('fp:  null标签预测错误的个数')

    print('---')
    # print('df_list:', df_list)
    print('avg df:', sum(df_list) / df_line)
    print('---end')

if __name__ == "__main__":
    # assert len(sys.argv) == 3
    # pin, pout = sys.argv[1:3]
    step = 0
    while step < 150000:
    # while step < 5000:
        step += 5000
        print('step ', step)
        # pin = 'cl/out/model_step_' + str(step) + '.pt/tl_test_tgt.txt'
        # pout = 'cl/test_tgt_bpe.txt'
        pin = 'cl/out/model_step_' + str(step) + '.pt/tl_test_tgt.txt.post'
        pout = 'cl/test_tgt.txt'
        judge(pin, pout)

"""
A B C D
A C B D
"""

