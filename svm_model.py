import numpy as np

class Support_Vector_Machine:
    #def __init__(self):
    # 	train
    def fit(self, data, nbr_feat):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        #nbr_feat = 3
        self.nbr_feat = nbr_feat
        #transforms = np.random.random((4, self.nbr_feat))

        #for i in range(4):
        #    for j in range(self.nbr_feat):
        #        if transforms[i, j] <= 0.5:
        #            transforms[i,j] = -1
        #        else:
        #            transforms[i,j] = 1
        #print(transforms)

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1, -1]
                      ]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi.w+b) = 1
        

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,
                      ]

        
        
        # extremely expensive
        b_range_multiple = 2
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum, latest_optimum])
            #print(w)
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        #print("w_t", w_t)
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        # 
                        # #### add a break here later..
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    #print(xi,':',yi*(np.dot(w_t,xi)+b))
                                    
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                            #print(opt_dict)

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            #print("norms", norms)
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                #print("yi", yi)
                print(xi,':',yi*(np.dot(self.w,xi)+self.b))            

    def predict(self,features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        #if classification !=0 and self.visualization:
        #    self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        
        #print(np.sum(self.w))
        #print("w", self.w)
        print("class", features, classification)
        return classification
        

        
data_dict = {-1:np.array([[1,7, 3],
                          [2,8, 4],
                          [1, 1, 1],
                          [3,8, 5],
                          [2,9,0]
                          ]),
             
             1:np.array([[5,1, 2],
                         [6,-1, 1],
                         [-1, -1, -1],
                         [7,3, 4],
                         [-1, -2, -9]
                        ])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict, nbr_feat=3)

predict_us = [[0,-10, -12],
              [1,3, 4],
              [3,4, 7],
              [3,5, 4],
              [5,5, 4],
              [5,6, 1],
              [6,-5, 2],
              [5,8, 5]]

for p in predict_us:
    svm.predict(p)


