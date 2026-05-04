
import { RouterProvider } from 'react-router-dom'
import './App.css'
import { router } from './router/router'
import Header from './components/ui/header'

function App() {

  return (
    <>
      <Header/>
      <RouterProvider router={router} />
    </>
  )
}

export default App
