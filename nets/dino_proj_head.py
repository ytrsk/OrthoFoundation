from lightly.models.modules.heads import DINOProjectionHead
class DINOProjHead(DINOProjectionHead):

    def __init__(self, **kwargs):
        super(DINOProjHead, self).__init__(1024, 2048, 256, batch_norm=False, **kwargs)
    def forward(self, x):
        x = x[-1]
        x = x[:, 0]
        return super().forward(x)   
        
def get_model(**kwargs):
    return DINOProjHead(**kwargs)