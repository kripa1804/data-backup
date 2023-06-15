import streamlit as st
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '%.4f' % x)


st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('C:/Users/325499/Downloads/vgg19_stage2_fc-0.610 (1).hdf5')
  
  return model
with st.spinner('Part Model is being loaded..'):
  model=load_model()


def load_model2():
  model2=tf.keras.models.load_model('C:/Users/325499/Desktop/damage detection/vgg19_stage1_fc-0.907.hdf5')
  #model2=tf.keras.models.load_model('C:/Users/dis531/Downloads/CAR_DAMAGE/Vgg19/vgg19_stage1_fc-0.967.hdf5')
  return model2
with st.spinner('Base Model is being loaded..'):
  model2=load_model2()


st.write("""
         # Car Parts Detection
         """
         )

file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):

        size = (256,256)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis]
    
        prediction = model.predict(img_reshape)
        
        return prediction

def import_and_predict2(image_data, model2):

        size = (256,256)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis]
        
        prediction2 = model2.predict(img_reshape)
        if prediction2 <=0.5:
          print("Damage")
          
        else:
          print('Not Damaged')
          
        return prediction2
    
if file is None:
    st.text("Please upload an image file")    
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    predictions2 = import_and_predict2(image,model2)
    class_name = ['Damage','No Damage']
    if predictions2 > 0.5:
        st.text("the car is not damaged")
    else:
        class_name = ['Damage','No Damage']
        class_names = ['bumper_dent','bumper_scratch','crash','door_dent','door_scratch','glass_shatter','head_lamp','tail_lamp']
        string = 'This car is '+class_name[np.argmax(predictions2)] +class_names[np.argmax(predictions)]
        st.success(string)
        #st.success(class_name[np.argmax(predictions2)])
        df = pd.DataFrame({ 'Damage/No Damage': [class_name[np.argmax(predictions2)]],'Damaged_part':[class_names[np.argmax(predictions)]]})
        conditions = [
        (df['Damaged_part'] == 'bumper_dent') | (df['Damaged_part'] == 'door_dent'),
        (df['Damaged_part'] == 'bumper_scratch') | (df['Damaged_part'] == 'door_scratch'),
        (df['Damaged_part'] == 'glass_shatter') | (df['Damaged_part'] == 'head_lamp') | (df['Damaged_part'] == 'tail_lamp'),
        (df['Damaged_part'] == 'crash')]
        choices = ['moderate', 'minor', 'minor','severe']
        df['severity'] = np.select(conditions, choices, default=None)
        conditions = [
        (df['Damaged_part'] == 'bumper_dent') | (df['Damaged_part'] == 'door_dent') | (df['Damaged_part'] == 'bumper_scratch') | (df['Damaged_part'] == 'door_scratch'),
        (df['Damaged_part'] == 'glass_shatter') | (df['Damaged_part'] == 'head_lamp') | (df['Damaged_part'] == 'tail_lamp') | (df['Damaged_part'] == 'crash')]
        choices = ['repair', 'replace']
        df['action'] = np.select(conditions, choices, default=None)
        #df=df.pivot_table(index=['sl no'],columns=['Damage/No Damage','Damaged_part','severity'])
        st.table(df)
        #stage2 dataframe
        class_names = ['bumper_dent','bumper_scratch','crash','door_dent','door_scratch','glass_shatter','head_lamp','tail_lamp']

        classification = np.argmax(predictions)

        #(df['Damaged_part'] == 'bumper_scratch') | (df['Damaged_part'] == 'door_scratch'),
        final = pd.DataFrame({'name' :np.array(class_names) ,'probability' :predictions[0]})

        final.sort_values(by = 'probability',ascending=False),class_names[classification]

        st.table(final)







