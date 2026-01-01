import numpy as np

class TransformDistribuition():
    
    def __init__(self, transform = lambda x:x, 
                       inv_transform = lambda x:x, 
                       jacobian_transform = lambda x:1,
                       transform_data = True,
                       transform_position = False,
                       transform_scale = False):
        super().__init__()
        self.transform = transform
        self.inv_transform = inv_transform
        self.jacobian_transform = jacobian_transform
        self.transform_data = transform_data
        self.transform_position = transform_position
        self.transform_scale = transform_scale


    def fit(self,xs,method='mvs'):
        self.fit_to = xs
        if self.transform_data:
            self.fit_to = self.transform(xs) # Transforma os dados de entrada

        super().fit(self.fit_to, method)  # Ajusta a distribuição aos dados transformados
        
        if self.transform_position:
            position = self.transform(self.position)
            self.set_position(position)
            self.scipy_params['loc'] = position
        
        if self.transform_scale:
            scale = self.transform(self.scale)
            self.set_scale(scale)
            self.scipy_params['scale'] = scale

        self.fit_to = xs    #Retorna os dados de entrada ao normal
        self.update_plot_properties()  #Atualiza as propriedades de plotagem para os dados de entrada não transformados


    def pdf(self, x):
        if self.transform_data:
            x_trans = self.transform(x)
            pdf_base = super().pdf(x_trans)
            jacobian = np.abs(self.jacobian_transform(x))
            return pdf_base * jacobian
        return super().pdf(x)
        
    def cdf(self, x):
        if self.transform_data:
            return super().cdf(self.transform(x))
        return super().cdf(x)
        
    def ppf(self, q):
        if self.transform_data:
            return self.inv_transform(super().ppf(q))
        super().ppf(q)
    