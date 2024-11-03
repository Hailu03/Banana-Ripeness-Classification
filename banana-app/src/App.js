import React, { useEffect, useState } from 'react';
import { View, Button, Image, Text } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { Camera } from 'expo-camera';

const App = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);

  const pickImage = async () => {
    // Request permission to access the media library
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (permissionResult.granted === false) {
      alert("Permission to access camera roll is required!");
      return;
    }

    // Launch image picker
    const pickerResult = await ImagePicker.launchImageLibraryAsync();

    if (pickerResult.cancelled) {
      console.log('User cancelled image picker');
      return;
    } else {
      setImage({ uri: pickerResult.assets[0].uri });
      sendImage(pickerResult.assets[0]);
    }
  };

  const sendImage = async (asset) => {
    const uri = asset.uri;
    const base64Image = await convertToBase64(uri);

    try {
      const response = await axios.post('http://<your-ip-adress>:<your-port>/predict', {
        image: `data:image/jpeg;base64,${base64Image}`
      });
      console.log('Response:', response.data);
      setResult(response.data.class);
    } catch (error) {
      console.error('Error sending image:', error);
      setResult(null);
    }
  };

  const convertToBase64 = async (uri) => {
    const response = await fetch(uri);
    const blob = await response.blob();
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        resolve(reader.result.split(',')[1]);
      };
      reader.readAsDataURL(blob);
    });
  };

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Button title="Pick an image from camera roll" onPress={pickImage} />
      {image && (
        <Image
          source={{ uri: image.uri }}
          style={{ width: 300, height: 300, marginTop: 20 }}
        />
      )}
      {result !== null && (
        <Text style={{ marginTop: 20, fontSize: 18 }}>
          Prediction Result: {result}
        </Text>
      )}
    </View>
  );
};

export default App;
