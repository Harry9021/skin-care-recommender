import './App.css';
import Router from './Router/Router';
import { CartProvider } from "./components/context/CartContext";
import FormPage from './components/Formpage'; // Adjust the import path as necessary

function App() {
  return (
    <div className="App">
       <CartProvider>
            <Router />
        </CartProvider>
    </div>
  );
}

export default App;
