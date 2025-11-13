import argparse

from util import (
    DebateOneByOne,
    DebateOneByOneWithLogprob,
    DebateOneByOneWithSemanticEntropy,
    DebateOneByOneSelfElicit,
    DebateOneByOneWithClusterConf,
    DebateOneByOneInterventions,
    DebateOneByOneMultiPersona,
    DebateSimultaneousSelfElicit,
    DebateSimultaneousChatEval,
    TemperatureScaling
)


def main(args):
    if not args.debate_conf:
        if args.intervention:
            DebateOneByOneInterventions(args)
        elif args.multi_persona:
            DebateOneByOneMultiPersona(args)
        elif args.chateval:
            DebateSimultaneousChatEval(args)
        else:
            DebateOneByOne(args)
            
    else:
        if args.conf_mode == 'length_norm' or args.conf_mode == 'seq_prob':
            DebateOneByOneWithLogprob(args)
        if args.conf_mode == 'semantic_entropy':
            DebateOneByOneWithSemanticEntropy(args)
        if args.conf_mode == 'self_elicit':
            if args.simultaneous:
                DebateSimultaneousSelfElicit(args)
            else:
                DebateOneByOneSelfElicit(args)
        if args.conf_mode == 'cluster':
            DebateOneByOneWithClusterConf(args)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM debating with confidence')
    parser.add_argument('--task', type=str, default='BBH', choices=['GSM', 'StrategyQA', 'HotpotQA', 'SQuAD', 'MMLU', 'SciQ', 'BBH', "CMS", "BIGGSM", "MATH", "GPQA", "MMLUPRO"], help='This argument assigns the task to be performed by')
    parser.add_argument('--debate_turns', type=int, default=3, help='This argument assigns the debating turns')
    parser.add_argument('--debate_agents', type=int, default=2, help='This argument assigns the number of debating agents')
    parser.add_argument('--debate_mode', type=str, default='onebyone', choices=['onebyone', 'simultaneous'], help='This argument assigns the debating mode, using only assistant_1 in this mode')
    parser.add_argument('--debate_conf', action='store_true', help='This argument decides whether to use confidence score in debating')
    parser.add_argument('--single', action='store_true', help='Single mode, no debate')
    parser.add_argument('--cot', action='store_true', help='Single mode, no debate')
    parser.add_argument('--intervention', action='store_true', help='Debate with interventions')
    parser.add_argument('--multi_persona', action='store_true', help='Debate with multi persona')
    parser.add_argument('--chateval', action='store_true', help='Debate with chateval')
    parser.add_argument('--simultaneous', action='store_true', help='Simultaneous debate')
    
    
    parser.add_argument('--calibration', action='store_true', help='This argument decides whether to calibrate the confidence score when debating. Note that "--calibration_train" and "--calibration" cannot be True at the same time')
    parser.add_argument('--calibration_train', action='store_true', help='This argument decides whether to train the calibration model. Note that "--calibration_train" and "--calibration" cannot be True at the same time')
    parser.add_argument('--calibration_scheme', type=str, default='temperature', choices=['histogram', 'isotonic', 'platt', 'temperature'], help='This argument decides which method will be used to calibrate the confidence score')
    parser.add_argument('--calibration_conf', type=str, default=None, help='This argument determines the confidence mode corresponding to the calibration model used to calibrate the confidence score.')
    parser.add_argument('--calibration_task', type=str, default=None, help='This argument determines the task corresponding to the calibration model used to calibrate the confidence score.')
    parser.add_argument('--calibration_overwrite', action='store_true', help='This argument determines the whether to overwrite the existing calibration model')
    parser.add_argument('--categorical', action='store_true', help='Activate categorical confidence scoring')
    parser.add_argument('--categorical_bins', type=int, default=10, help='Categorical confidence scoring bins')
    parser.add_argument('--top_logprobs', type=int, default=10, help='This argument assigns the top N tokens\' logprobs for temperature scaling')
    parser.add_argument('--low_index_calibration', type=int, default=0, help='This argument indicates the lower bound of the dataset used to train calibration model')
    parser.add_argument('--up_index_calibration', type=int, default=1000, help='This argument indicates the upper bound of the dataset used to train calibration model')
    
    parser.add_argument('--attempt_times', type=int, default=2, help='This argument assigns the number of attempts times if failed to output in requested format')
    
    parser.add_argument('--conf_mode', type=str, default='length_norm', choices=['self_elicit', 'length_norm', 'seq_prob', 'semantic_entropy', 'random', 'cluster'], help='This argument assigns the method use for extratcing confidence')
    parser.add_argument('--conf_type', type=str, default='score', choices=['score', 'level'], help='This argument assigns the type of confidence score')
    parser.add_argument('--cluster_sample_times', type=int, default=10, help='This argument assigns how many times to sample when clustering')
    parser.add_argument('--output_dir', type=str, default='./result', help='This argument assigns the output directory')
    
    parser.add_argument('--low_index', type=int, default=0, help='This argument indicates the lower bound of the dataset used')
    parser.add_argument('--up_index', type=int, default=2000, help='This argument indicates the upper bound of the dataset used')
    parser.add_argument('--save_interval', type=int, default=10, help='This argument indicates the save interval when debating, i.e. the granularity when saving data')
    parser.add_argument('--num_workers', type=int, default=5, help='This argument indicates the number of concurrent workers when debating')
    
    args = parser.parse_args()
    
    if args.calibration and args.calibration_train:
        raise ValueError("'--calibration' and '--calibration_train' cannot be true at the same time.")
        
    if args.calibration_train and not args.debate_conf:
        raise ValueError("'--calibration_train' must be used with '--debate_conf'.")
    
    if args.calibration and not args.debate_conf:
        raise ValueError("'--calibration' must be used with '--debate_conf'.")
        
    main(args)