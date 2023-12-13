{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: keras in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (2.14.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install keras"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pip in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (23.3.1)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install --upgrade pip"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: opencv-python in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (4.8.1.78)\n",
      "Requirement already satisfied: numpy>=1.21.2 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from opencv-python) (1.23.5)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install opencv-python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: tensorflow in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (2.14.0)\n",
      "Requirement already satisfied: tensorflow-intel==2.14.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow) (2.14.0)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (2.0.0)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.6.3)\n",
      "Requirement already satisfied: flatbuffers>=23.5.26 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (23.5.26)\n",
      "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (0.5.4)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (0.2.0)\n",
      "Requirement already satisfied: h5py>=2.9.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (3.7.0)\n",
      "Requirement already satisfied: libclang>=13.0.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (16.0.6)\n",
      "Requirement already satisfied: ml-dtypes==0.2.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (0.2.0)\n",
      "Requirement already satisfied: numpy>=1.23.5 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.23.5)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (3.3.0)\n",
      "Requirement already satisfied: packaging in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (22.0)\n",
      "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (4.24.4)\n",
      "Requirement already satisfied: setuptools in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (65.6.3)\n",
      "Requirement already satisfied: six>=1.12.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.16.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (2.3.0)\n",
      "Requirement already satisfied: typing-extensions>=3.6.6 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (4.4.0)\n",
      "Requirement already satisfied: wrapt<1.15,>=1.11.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.14.1)\n",
      "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (0.31.0)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.59.0)\n",
      "Requirement already satisfied: tensorboard<2.15,>=2.14 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (2.14.1)\n",
      "Requirement already satisfied: tensorflow-estimator<2.15,>=2.14.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (2.14.0)\n",
      "Requirement already satisfied: keras<2.15,>=2.14.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.14.0->tensorflow) (2.14.0)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.14.0->tensorflow) (0.38.4)\n",
      "Requirement already satisfied: google-auth<3,>=1.6.3 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.23.3)\n",
      "Requirement already satisfied: google-auth-oauthlib<1.1,>=0.5 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (1.0.0)\n",
      "Requirement already satisfied: markdown>=2.6.8 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.4.1)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.28.1)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (0.7.1)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.2.2)\n",
      "Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (5.3.1)\n",
      "Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (0.2.8)\n",
      "Requirement already satisfied: rsa<5,>=3.1.4 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (4.9)\n",
      "Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from google-auth-oauthlib<1.1,>=0.5->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (1.3.1)\n",
      "Requirement already satisfied: charset-normalizer<3,>=2 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.4)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (1.26.14)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2022.12.7)\n",
      "Requirement already satisfied: MarkupSafe>=2.1.1 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from werkzeug>=1.0.1->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.1.1)\n",
      "Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (0.4.8)\n",
      "Requirement already satisfied: oauthlib>=3.0.0 in c:\\users\\prince kumar\\anaconda3\\lib\\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.2.2)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install tensorflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import load_model\n",
    "import time\n",
    "import cv2\n",
    "import numpy as np\n",
    "import sys"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'none'}\n"
     ]
    }
   ],
   "source": [
    "REV_CLASS_MAP = {}\n",
    "c = 0\n",
    "for i in range(65,91):\n",
    "        REV_CLASS_MAP[c] = chr(i)\n",
    "        c += 1\n",
    "REV_CLASS_MAP[26] = 'none'\n",
    "print(REV_CLASS_MAP)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 322ms/step\n",
      "Predicted: A\n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "# Define a function to map sign codes to their corresponding names (if available).\n",
    "def mapper(val):\n",
    "    return REV_CLASS_MAP[val]  # Define REV_CLASS_MAP as needed\n",
    "\n",
    "# Load the model.\n",
    "model = load_model(\"asl-model2.h5\")\n",
    "\n",
    "# Define the file path for the image.\n",
    "filepath = './asl_alphabet_test/asl_alphabet_test/A_test.jpg'  # Use double backslashes or raw string 'r' for Windows paths\n",
    "\n",
    "# Load the image and perform checks.\n",
    "img = cv2.imread(filepath)\n",
    "\n",
    "if img is not None:\n",
    "    if img.shape[:2] != (0, 0):\n",
    "        # Resize the image to the desired dimensions.\n",
    "        img = cv2.resize(img, (227, 227))\n",
    "\n",
    "        # Predict the sign made.\n",
    "        pred = model.predict(np.array([img]))\n",
    "        sign_code = np.argmax(pred[0])\n",
    "        sign_name = mapper(sign_code)\n",
    "\n",
    "        print(\"Predicted: {}\".format(sign_name))\n",
    "    else:\n",
    "        print(\"Error: Image dimensions are empty.\")\n",
    "else:\n",
    "    print(\"Error: Failed to load the image.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:5 out of the last 5 calls to <function Model.make_predict_function.<locals>.predict_function at 0x0000028AB2E39E10> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.\n",
      "1/1 [==============================] - 0s 318ms/step\n",
      "[[2.1911574e-18 1.0532195e-18 3.9681770e-16 4.5017981e-18 1.9845663e-18\n",
      "  8.1956129e-16 8.8346197e-12 1.0000000e+00 1.4447075e-18 8.9874018e-11\n",
      "  1.7893941e-18 2.2935382e-18 3.2911150e-16 8.7301059e-17 1.0007202e-17\n",
      "  1.5560873e-12 5.9557062e-18 2.8296499e-16 8.1020118e-17 1.2964375e-16\n",
      "  1.8731956e-18 4.6007408e-17 4.3287755e-18 1.3167573e-17 2.8853373e-17\n",
      "  1.2368705e-17 2.8394450e-17]]\n",
      "Predicted: H\n"
     ]
    }
   ],
   "source": [
    "filepath = './asl_alphabet_test/asl_alphabet_test/H_test.jpg'\n",
    "\n",
    "\n",
    "def mapper(val):\n",
    "    return REV_CLASS_MAP[val]\n",
    "\n",
    "\n",
    "model = load_model(\"asl-model2.h5\")\n",
    "\n",
    "# prepare the image\n",
    "img = cv2.imread(filepath)\n",
    "#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "img = cv2.resize(img, (227, 227))\n",
    "\n",
    "# predict the sign made\n",
    "pred = model.predict(np.array([img]))\n",
    "sign_code = np.argmax(pred[0])\n",
    "sign_name = mapper(sign_code)\n",
    "\n",
    "print(pred)\n",
    "print(\"Predicted: {}\".format(sign_name))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mapper(val):\n",
    "    return REV_CLASS_MAP[val]\n",
    "\n",
    "model = load_model(\"asl-model2.h5\")\n",
    "\n",
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "text_msg = ''\n",
    "\n",
    "while True:\n",
    "    ret, frame = cap.read()\n",
    "    if not ret:\n",
    "        continue\n",
    "\n",
    "    \n",
    "    cv2.rectangle(frame, (10, 10), (310, 310), (255, 255, 255), 2)\n",
    "    \n",
    "    # extract the region of image within the user rectangle\n",
    "    roi = frame[10:310, 10:310]\n",
    "    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)\n",
    "    img = cv2.resize(img, (227, 227))\n",
    "\n",
    "    # predict the sign made\n",
    "    pred = model.predict(np.array([img]))\n",
    "    sign_code = np.argmax(pred[0])\n",
    "    sign_name = mapper(sign_code)\n",
    "    \n",
    "    text_msg += sign_name\n",
    "    \n",
    "    font = cv2.FONT_HERSHEY_SIMPLEX\n",
    "    cv2.putText(frame, \"Your Sign: \" + sign_name,\n",
    "                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)\n",
    "    \n",
    "    cv2.putText(frame, \"Text Message: \" + text_msg,\n",
    "                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)\n",
    "    \n",
    "    cv2.imshow(\"ASL Sign to Text\", frame)\n",
    "\n",
    "    k = cv2.waitKey(10)\n",
    "    if k == ord('q'):\n",
    "        break\n",
    "    #putting identified text into a text file\n",
    "    def text_add():\n",
    "        f=open(\"output.txt\", mode='w', encoding='utf-8')\n",
    "        if text_msg != 'none':\n",
    "            f.write(text_msg)\n",
    "    text_add()\n",
    "        \n",
    "f.close()\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()\n",
    "\n",
    "file = open('output.txt')\n",
    "l = file.readlines()\n",
    "\n",
    "# filtering through the output message\n",
    "\n",
    "print(l)\n",
    "f2 = open('output_modified.txt', mode='w', encoding='utf-8')\n",
    "lst = []\n",
    "start_time = time.time()  # Get the current time\n",
    "\n",
    "while time.time() - start_time < 5:  # Run for 5 seconds\n",
    "    for i in l[0]:\n",
    "        lst.append(i)\n",
    "        time.sleep(3)\n",
    "print(lst)\n",
    "lst2 = []\n",
    "c = 0\n",
    "while time.time() - start_time < 5:\n",
    "    for i in range(len(lst)):\n",
    "        if i == 0:\n",
    "            sign_prev = ''\n",
    "        else:\n",
    "            sign_prev = lst[i-1]\n",
    "        sign_present = lst[i]\n",
    "        if sign_present == sign_prev:\n",
    "            c += 1\n",
    "            if c>8:\n",
    "                if sign_present not in lst2:\n",
    "                    lst2.append(sign_present)\n",
    "                    time.sleep(3)\n",
    "            else:\n",
    "                continue\n",
    "        else:\n",
    "            c = 0\n",
    "            continue\n",
    "    print(lst2)\n",
    "\n",
    "msg = ''\n",
    "for i in lst2:\n",
    "    msg+=i\n",
    "f2.write(msg)\n",
    "        \n",
    "f2.close()\n",
    "with open('output.txt', 'r', encoding='utf-8') as file:\n",
    "    data = file.read()\n",
    "unique_data = [data[0]] + [data[i] for i in range(1, len(data)) if data[i]!=data[i-1]]\n",
    "with open('output_modified.txt', 'w', encoding='utf-8') as f2:\n",
    "    f2.write(''.join(unique_data))\n",
    "\n",
    "f2.close()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
