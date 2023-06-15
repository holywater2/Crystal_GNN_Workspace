import os
import sys
import wandb
import datetime
sys.path.append('..')
sys.path.append('/home/holywater2/2023/_Reproduce')

def wandb_init_and_output_dir(args, config):
    wandb_mode = None
    if args['wandb_disabled']:
        wandb_mode = "disabled"
        
    wandb.init(project=args["project"], 
            config=config,
            mode=wandb_mode,
            allow_val_change=True)

    for key in wandb.config.keys():
        args[key] = config[key]
    
    inp_config = config
    config = wandb.config
    
    if args['data_pdirname'] is None:
        args['data_pdirname'] = "../dataset/mp_megnet"
    
    datestr = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    args['output_dir'] = './results/' + config['model'] + '_' \
                + config['dataset'] + '_' \
                + config['prop'] + '_' \
                + datestr
    
    if args["mode"] == "sample":
        args['output_dir'] = args['output_dir'] + "_sample"
    
    print("[I] Output directory is",args['output_dir'])
    os.makedirs(args['output_dir'])

    return args, config, inp_config