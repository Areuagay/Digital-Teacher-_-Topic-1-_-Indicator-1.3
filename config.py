import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_train',type=bool, default=True) # False
    parser.add_argument('--loss', default='SMRRLoss', help='BCELoss/SMRRLoss')
    parser.add_argument('--model', default='HGGNN', help='SRGNN/HGGNN')
    parser.add_argument('--data', default='reddit', help='xing/reddit')
    parser.add_argument('--max_session', type=int, default=30) # Xing: 50, Reddit: 30.
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--sample_size', type=int, default=12)

    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--gnn_layer_size', type=int, default=2)

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--learning_rate', type=float, default=0.0005, # 0.0005
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5) #1e-5
    parser.add_argument('--batch_size', type=int, default=512) #512
    parser.add_argument('--epoch', type=int, default=100) #100
    parser.add_argument('--patience', type=int, default=3)#15
    parser.add_argument('--dropout', type=float, default=0.7) #  probability of an element to be zeroed.
    parser.add_argument('--lr_dc_step', type=int, default=3) #3
    parser.add_argument('--lr_dc', type=float, default=0.1)

    return parser.parse_args()