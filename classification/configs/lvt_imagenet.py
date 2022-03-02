config = dict(
    # classification/downstream tasks
    with_cls_head = True,
    
    # rasa setting
    rasa_cfg = dict(
        atrous_rates= [1,3,5], # None, [1,3,5]
        act_layer= 'nn.SiLU(True)',
        init= 'kaiming',
        r_num = 2,
    ),
)
