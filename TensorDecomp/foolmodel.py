from foolbox.adversarial import Adversarial
from foolbox.models import PyTorchModel
from foolbox.attacks import DeepFoolAttack
from models import get_network
from datasets import get_DataLoader
from TensorDecomp.config import config
CUDA_LAUNCH_BLOCKING=1

if __name__ == "__main__":
    
    base = get_network()
    base.load_state_dict(torch.load(os.path.join(config.MODEL.CKPT_ROOT,
                                    config.DATASET.NAME, 'Undecomposed', config.MODEL.NAME+'.pth')))

    decomp = copy.deepcopy(base)
    decomp = decompose(decomp)
    decomp.load_state_dict(torch.load(os.path.join(config.MODEL.CKPT_ROOT,
                                    config.DATASET.NAME, 'decomposed',
                                    config.MODEL.NAME+'_'+config.SOLVER.LOSS+'.pth')))

    base.eval()
    decomp.eval()

    print("Loading Data-sets .....")
    _, test_loader = get_DataLoader()
    print("Data-sets loaded")

    base_wrapper = PyTorchModel(base, (0,1), config.DATASET.NUM_CLASSES)
    decomp_wrapper = PyTorchModel(decomp, (0,1), config.DATASET.NUM_CLASSES)

    if config.ATTACK.NAME.lower() == 'pgd':
        attack_base = PGD(base_wrapper, distance=foolbox.distaces.Linfinity)
        attack_decomp = PGD(decomp_wrapper, distance=foolbox.distaces.Linfinity)
    else if config.ATTACK.NAME.lower() == 'cw':
        attack_base = CarliniWagnerL2Attack(base_wrapper)
        attack_decomp = CarliniWagnerL2Attack(decomp_wrapper)
    else if config.ATTACK.NAME.lower() == 'deepfool':
        attack_base = DeepFoolAttack(base_wrapper)
        attack_decomp = DeepFoolAttack(decomp_wrapper)
    else:
        raise NotImplementedError("{} attack has not been implemented yet.".format(config.ATTACK.NAME))

    r_base_tot = 0.
    r_decomp_tot = 0.
    correct_base = 0
    correct_decomp = 0
    eps = 0.01
    for batch_idx, (image, label) in enumerate(test_loader):
        image, label = image.numpy(),label.numpy()
        
        if config.ATTACK.NAME.lower() == 'pgd':
            perturb_base = attack_base(image, label, epsilon=config.ATTACK.EPS)
            perturb_decomp = attack_decomp(image, label, epsilon=config.ATTACK.EPS)
        
        else if config.ATTACK.NAME.lower() == 'cw':
            perturb_base = attack_base(image, label, 
                                        max_iterations=config.ATTACK.MAX_ITER,
                                        learning_rate=config.ATTACK.LR)
            perturb_decomp = attack_decomp(image, label, 
                                            max_iterations=config.ATTACK.MAX_ITER, 
                                            learning_rate=config.ATTACK.LR)

        else:
            perturb_base = attack_base(image, label)
            perturb_decomp = attack_decomp(image, label)
    
        r_base_tot += np.linalg.norm(perturb_base - image) / np.linalg.norm(image)
        r_decomp_tot += np.linalg.norm(perturb_decomp - image) / np.linalg.norm(image)

        if batch_idx % 100 == 0 and batch_idx>0:
            print("{} :  robustness_base - {} ;  robustness_decomp - {}".format(batch_idx, 
                                                                        r_base_tot/(batch_idx+1),
                                                                       r_decomp_tot/(batch_idx+1)))
    
    print('Baseline robustness:         {}'.format(r_base_tot/batch_idx))
    print('Decomposed_{} robustness:    {}'.format(r_decomp_tot/batch_idx))
