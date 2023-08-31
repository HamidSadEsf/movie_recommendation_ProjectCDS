# Preprocess and save data for collaborative filtering modelling
# Uncomment only when you want to (re)build your ratings matrix from scratch

#import data_script.preprocess_collaborative

# Train collaborative filtering models, cross validate moleds, save the models to disk
# Uncomment the prefered model(s)
import model.collaborative_filtering_models.KnnBasic
#import model.collaborative_filtering_modeles.KnnWithMeans

