import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import DemoPage from './pages/DemoPage'
import EmotionalTTSPage from './pages/EmotionalTTSPage'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/emotional-tts" element={<EmotionalTTSPage />} />
      </Routes>
    </Router>
  )
}

export default App
