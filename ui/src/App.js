import './App.css';
import Router from './Router/Router';
import { CartProvider } from "./components/context/CartContext";

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
