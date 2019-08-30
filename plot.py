import matplotlib.pyplot as plt

# Plot results
def draw_label(labelSet, prediction, xRange):
  plt.subplot(2, 1, 1)
  plt.plot(range(1, labelSet.shape[0]+1), labelSet, "r")
  plt.xlim(xRange)
  plt.subplot(2, 1, 2)
  plt.plot(range(1, labelSet.shape[0]+1), prediction, "b")
  plt.xlim(xRange)
  plt.show()

def draw_coloredLabel(labelSet, prediction, xRange):
  plt.subplot(2, 1, 1)

  label_flag = labelSet[0]
  segment = [labelSet[0]]
  segment_flag = 1
  talking_count = 0
  opening_count = 0
  chewing_count = 0
  swallowing_count = 0
  other_count = 0

  for i in range(0,labelSet.shape[0]):
      # print(i)
      if label_flag == labelSet[i]:
         segment.append(labelSet[i]) 
      else:
         if label_flag == 0:
            talking_count = talking_count+1
            if talking_count == 1:
               plt.plot(range(segment_flag-1,i+1),segment,'r',label='Talking')
            else:
               plt.plot(range(segment_flag-1,i+1),segment,'r')
         if label_flag == 1:
            opening_count = opening_count+1
            if opening_count == 1:
               plt.plot(range(segment_flag-1,i+1),segment,'g',label='Opening')
            else:
               plt.plot(range(segment_flag-1,i+1),segment,'g')
         if label_flag == 2:
            chewing_count = chewing_count+1
            if chewing_count == 1:
               plt.plot(range(segment_flag-1,i+1),segment,'b',label='Chewing')
            else:
               plt.plot(range(segment_flag-1,i+1),segment,'b')
         if label_flag == 3:
            swallowing_count = swallowing_count+1
            if swallowing_count == 1:
               plt.plot(range(segment_flag-1,i+1),segment,'orange',label='Swallowing')   
            else:
               plt.plot(range(segment_flag-1,i+1),segment,'orange')       
         if label_flag == 4:
            other_count = other_count+1
            if other_count == 1:
               plt.plot(range(segment_flag-1,i+1),segment,'yellow',label='Other')   
            else:
               plt.plot(range(segment_flag-1,i+1),segment,'yellow')     
         segment = []
         segment.append(labelSet[i-1])
         segment.append(labelSet[i])
         segment_flag = i+1
      label_flag = labelSet[i]

      if i == labelSet.shape[0]-1:
         if label_flag == 0:
            talking_count = talking_count+1
            if talking_count == 1:
               plt.plot(range(segment_flag-1,i+2),segment,'r',label='Talking')
            else:
               plt.plot(range(segment_flag-1,i+2),segment,'r')
         if label_flag == 1:
            opening_count = opening_count+1
            if opening_count == 1:
               plt.plot(range(segment_flag-1,i+2),segment,'g',label='Opening')
            else:
               plt.plot(range(segment_flag-1,i+2),segment,'g')
         if label_flag == 2:
            chewing_count = chewing_count+1
            if chewing_count == 1:
               plt.plot(range(segment_flag-1,i+2),segment,'b',label='Chewing')
            else:
               plt.plot(range(segment_flag-1,i+2),segment,'b')
         if label_flag == 3:
            swallowing_count = swallowing_count+1
            if swallowing_count == 1:
               plt.plot(range(segment_flag-1,i+2),segment,'orange',label='Swallowing')   
            else:
               plt.plot(range(segment_flag-1,i+2),segment,'orange')       
         if label_flag == 4:
            other_count = other_count+1
            if other_count == 1:
               plt.plot(range(segment_flag-1,i+2),segment,'yellow',label='Other')   
            else:
               plt.plot(range(segment_flag-1,i+2),segment,'yellow') 
  plt.legend()    
  plt.xlim(xRange)

  plt.subplot(2, 1, 2)

  label_flag = prediction[0]
  segment = [prediction[0]]
  segment_flag = 1
  talking_count = 0
  opening_count = 0
  chewing_count = 0
  swallowing_count = 0
  other_count = 0
  for i in range(0,prediction.shape[0]):
      if label_flag == prediction[i]:
         segment.append(prediction[i]) 
      else:
         if label_flag == 0:
            talking_count = talking_count+1
            if talking_count == 1:
               plt.plot(range(segment_flag-1,i+1),segment,'r',label='Talking')
            else:
               plt.plot(range(segment_flag-1,i+1),segment,'r')
         if label_flag == 1:
            opening_count = opening_count+1
            if opening_count == 1:
               plt.plot(range(segment_flag-1,i+1),segment,'g',label='Opening')
            else:
               plt.plot(range(segment_flag-1,i+1),segment,'g')
         if label_flag == 2:
            chewing_count = chewing_count+1
            if chewing_count == 1:
               plt.plot(range(segment_flag-1,i+1),segment,'b',label='Chewing')
            else:
               plt.plot(range(segment_flag-1,i+1),segment,'b')
         if label_flag == 3:
            swallowing_count = swallowing_count+1
            if swallowing_count == 1:
               plt.plot(range(segment_flag-1,i+1),segment,'orange',label='Swallowing')   
            else:
               plt.plot(range(segment_flag-1,i+1),segment,'orange')    
         if label_flag == 4:
            other_count = other_count+1
            if other_count == 1:
               plt.plot(range(segment_flag-1,i+1),segment,'yellow',label='Other')   
            else:
               plt.plot(range(segment_flag-1,i+1),segment,'yellow')     
         segment = []
         segment.append(prediction[i-1])
         segment.append(prediction[i])
         segment_flag = i+1
      label_flag = prediction[i]

      if i == prediction.shape[0]-1:
         if label_flag == 0:
            talking_count = talking_count+1
            if talking_count == 1:
               plt.plot(range(segment_flag-1,i+2),segment,'r',label='Talking')
            else:
               plt.plot(range(segment_flag-1,i+2),segment,'r')
         if label_flag == 1:
            opening_count = opening_count+1
            if opening_count == 1:
               plt.plot(range(segment_flag-1,i+2),segment,'g',label='Opening')
            else:
               plt.plot(range(segment_flag-1,i+2),segment,'g')
         if label_flag == 2:
            chewing_count = chewing_count+1
            if chewing_count == 1:
               plt.plot(range(segment_flag-1,i+2),segment,'b',label='Chewing')
            else:
               plt.plot(range(segment_flag-1,i+2),segment,'b')
         if label_flag == 3:
            swallowing_count = swallowing_count+1
            if swallowing_count == 1:
               plt.plot(range(segment_flag-1,i+2),segment,'orange',label='Swallowing')   
            else:
               plt.plot(range(segment_flag-1,i+2),segment,'orange')      
         if label_flag == 4:
            other_count = other_count+1
            if other_count == 1:
               plt.plot(range(segment_flag-1,i+2),segment,'yellow',label='Other')   
            else:
               plt.plot(range(segment_flag-1,i+2),segment,'yellow')   
  plt.legend() 
  plt.xlim(xRange)
  plt.show()

def draw_sensorData(sensorData, labelSet, prediction, xRange):
  plt.subplot(2, 1, 1)

  label_flag = labelSet[0]
  segment = []
  segment_flag = 1
  talking_count = 0
  opening_count = 0
  chewing_count = 0
  swallowing_count = 0
  other_count = 0

  for i in range(0,sensorData.shape[0]):
      # print(i)
      if label_flag == labelSet[i]:
         segment.append(sensorData[i]) 
      else:
         if label_flag == 0:
            talking_count = talking_count+1
            if talking_count == 1:
               plt.plot(range(segment_flag,i+1),segment,'r',label='Talking')
            else:
               plt.plot(range(segment_flag,i+1),segment,'r')
         if label_flag == 1:
            opening_count = opening_count+1
            if opening_count == 1:
               plt.plot(range(segment_flag,i+1),segment,'g',label='Opening')
            else:
               plt.plot(range(segment_flag,i+1),segment,'g')
         if label_flag == 2:
            chewing_count = chewing_count+1
            if chewing_count == 1:
               plt.plot(range(segment_flag,i+1),segment,'b',label='Chewing')
            else:
               plt.plot(range(segment_flag,i+1),segment,'b')
         if label_flag == 3:
            swallowing_count = swallowing_count+1
            if swallowing_count == 1:
               plt.plot(range(segment_flag,i+1),segment,'orange',label='Swallowing')   
            else:
               plt.plot(range(segment_flag,i+1),segment,'orange')       
         if label_flag == 4:
            other_count = other_count+1
            if other_count == 1:
               plt.plot(range(segment_flag,i+1),segment,'yellow',label='Other')   
            else:
               plt.plot(range(segment_flag,i+1),segment,'yellow')     
         segment = []
         # segment.append(cv_x[i-1])
         segment.append(sensorData[i])
         segment_flag = i+1
      label_flag = labelSet[i]

      if i == sensorData.shape[0]-1:
         if label_flag == 0:
            talking_count = talking_count+1
            if talking_count == 1:
               plt.plot(range(segment_flag,i+2),segment,'r',label='Talking')
            else:
               plt.plot(range(segment_flag,i+2),segment,'r')
         if label_flag == 1:
            opening_count = opening_count+1
            if opening_count == 1:
               plt.plot(range(segment_flag,i+2),segment,'g',label='Opening')
            else:
               plt.plot(range(segment_flag,i+2),segment,'g')
         if label_flag == 2:
            chewing_count = chewing_count+1
            if chewing_count == 1:
               plt.plot(range(segment_flag,i+2),segment,'b',label='Chewing')
            else:
               plt.plot(range(segment_flag,i+2),segment,'b')
         if label_flag == 3:
            swallowing_count = swallowing_count+1
            if swallowing_count == 1:
               plt.plot(range(segment_flag,i+2),segment,'orange',label='Swallowing')   
            else:
               plt.plot(range(segment_flag,i+2),segment,'orange')       
         if label_flag == 4:
            other_count = other_count+1
            if other_count == 1:
               plt.plot(range(segment_flag,i+2),segment,'yellow',label='Other')   
            else:
               plt.plot(range(segment_flag,i+2),segment,'yellow') 
  plt.legend()    
  plt.xlim(xRange)

  plt.subplot(2, 1, 2)

  label_flag = prediction[0]
  segment = []
  segment_flag = 1
  talking_count = 0
  opening_count = 0
  chewing_count = 0
  swallowing_count = 0
  other_count = 0
  for i in range(0,sensorData.shape[0]):
      if label_flag == prediction[i]:
         segment.append(sensorData[i]) 
      else:
         if label_flag == 0:
            talking_count = talking_count+1
            if talking_count == 1:
               plt.plot(range(segment_flag,i+1),segment,'r',label='Talking')
            else:
               plt.plot(range(segment_flag,i+1),segment,'r')
         if label_flag == 1:
            opening_count = opening_count+1
            if opening_count == 1:
               plt.plot(range(segment_flag,i+1),segment,'g',label='Opening')
            else:
               plt.plot(range(segment_flag,i+1),segment,'g')
         if label_flag == 2:
            chewing_count = chewing_count+1
            if chewing_count == 1:
               plt.plot(range(segment_flag,i+1),segment,'b',label='Chewing')
            else:
               plt.plot(range(segment_flag,i+1),segment,'b')
         if label_flag == 3:
            swallowing_count = swallowing_count+1
            if swallowing_count == 1:
               plt.plot(range(segment_flag,i+1),segment,'orange',label='Swallowing')   
            else:
               plt.plot(range(segment_flag,i+1),segment,'orange')    
         if label_flag == 4:
            other_count = other_count+1
            if other_count == 1:
               plt.plot(range(segment_flag,i+1),segment,'yellow',label='Other')   
            else:
               plt.plot(range(segment_flag,i+1),segment,'yellow')     
         segment = []
         # segment.append(cv_x[i-1])
         segment.append(sensorData[i])
         segment_flag = i+1
      label_flag = prediction[i]

      if i == sensorData.shape[0]-1:
         if label_flag == 0:
            talking_count = talking_count+1
            if talking_count == 1:
               plt.plot(range(segment_flag,i+2),segment,'r',label='Talking')
            else:
               plt.plot(range(segment_flag,i+2),segment,'r')
         if label_flag == 1:
            opening_count = opening_count+1
            if opening_count == 1:
               plt.plot(range(segment_flag,i+2),segment,'g',label='Opening')
            else:
               plt.plot(range(segment_flag,i+2),segment,'g')
         if label_flag == 2:
            chewing_count = chewing_count+1
            if chewing_count == 1:
               plt.plot(range(segment_flag,i+2),segment,'b',label='Chewing')
            else:
               plt.plot(range(segment_flag,i+2),segment,'b')
         if label_flag == 3:
            swallowing_count = swallowing_count+1
            if swallowing_count == 1:
               plt.plot(range(segment_flag,i+2),segment,'orange',label='Swallowing')   
            else:
               plt.plot(range(segment_flag,i+2),segment,'orange')      
         if label_flag == 4:
            other_count = other_count+1
            if other_count == 1:
               plt.plot(range(segment_flag,i+2),segment,'yellow',label='Other')   
            else:
               plt.plot(range(segment_flag,i+2),segment,'yellow')   
  plt.legend() 
  plt.xlim(xRange)
  plt.show()



