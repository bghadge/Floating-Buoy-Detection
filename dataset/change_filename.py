import os 
# from os.path import dirname
# Function to rename multiple files 
def main(): 
  
    for count, filename in enumerate(os.listdir("./labelled/1440_whitebuoy/")): 
        dst = "buoy_0000" + str(count) + ".png"
        src = './labelled/1440_whitebuoy/' + filename 
        dst = './labelled/1440_whitebuoy/' + dst 
        # rviz_screenshot_2020_06_29-17_10_53
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 