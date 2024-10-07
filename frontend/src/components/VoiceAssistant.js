// src/components/VoiceAssistant.js
import React, { useEffect } from 'react';
import { FaMicrophone } from 'react-icons/fa'; // Import React icon
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { useNavigate } from 'react-router-dom';

const VoiceAssistant = () => {
  const { transcript, resetTranscript } = useSpeechRecognition();
  const navigate = useNavigate();

  const commands = [
    {
      command: 'go to home',
      callback: () => navigate('/')
    },
    {
      command: 'your business',
      callback: () => navigate('/upload')
    },
    {
      command: 'virtual consultant',
      callback: () => navigate('/virtual-consultant')
    },
    {
      command: 'insight',
      callback: () => navigate('/insight-bot')
    },
    {
      command: 'dashboard',
      callback: () => navigate('/grouped-analysis')
    },
    {
      command: 'navigate to home',
      callback: () => navigate('/')
    },
    {
      command: 'navigate to your business',
      callback: () => navigate('/upload')
    },
    {
      command: 'navigate to virtual consultant',
      callback: () => navigate('/virtual-consultant')
    },
    {
      command: 'navigate to insight',
      callback: () => navigate('/insight-bot')
    },
    {
      command: 'navigate to dashboard',
      callback: () => navigate('/grouped-analysis')
    }
  ];

  useEffect(() => {
    const command = commands.find(cmd => cmd.command === transcript);
    console.log(command)
    if (command) {
      command.callback();
      resetTranscript(); // Clear the transcript after a command is executed
    }
  }, [transcript, commands, navigate, resetTranscript]);

  const startListening = () => {
    SpeechRecognition.startListening();
  };

  return (
    <div className="flex flex-col items-center">
      <button
        onClick={startListening}
        className="flex items-center justify-center p-2 rounded-full bg-blue-500 hover:bg-blue-600 transition duration-200"
        aria-label="Activate Voice Assistant"
      >
        <FaMicrophone className="text-white text-2xl" /> {/* React icon */}
      </button>
    </div>
  );
};

export default VoiceAssistant;
