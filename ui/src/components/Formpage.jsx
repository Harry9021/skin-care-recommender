import React, { useState } from "react";
import "../Styles/Formpage.css";
import image1 from "../Vectors/image5.png";
import image2 from "../Vectors/image6.png";
import { Link } from "react-router-dom";
import { IoMdArrowRoundBack } from "react-icons/io";

const Formpage = () => {
  const [formData, setFormData] = useState({
    product: "",
    skinType: "",
    concern1: "",
    concern2: "",
    concern3: "",
    keyIngredient: "",
    formulation: ""
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log(formData);
    alert("Form Submitted");
  };

  return (
    <div className="form-image-container">
      <div className="blue-box">
        <Link to="/">
        <div className="backarrow">
        <IoMdArrowRoundBack style={{width:30, height:30, marginLeft:10, marginTop:10}}/>
        </div>
        </Link>
      </div>
      <div className="form-container">
        <form onSubmit={handleSubmit}>
          <h3>Fill Below Information to know your skincare products.</h3>

          <label>Which product you want?</label>
          <select name="product" onChange={handleChange}>
            <option value="">Select</option>
            <option value="cleanser">Cleanser</option>
            <option value="moisturizer">Moisturizer</option>
            <option value="serum">Serum</option>
          </select>

          <label>What is your skin type?</label>
          <select name="skinType" onChange={handleChange}>
            <option value="">Select</option>
            <option value="dry">Dry</option>
            <option value="oily">Oily</option>
            <option value="combination">Combination</option>
            <option value="normal">Normal</option>
          </select>

          <label>Skin concern 1?</label>
          <select name="concern1" onChange={handleChange}>
            <option value="">Select</option>
            <option value="acne">Acne</option>
            <option value="wrinkles">Wrinkles</option>
            <option value="dark-spots">Dark Spots</option>
          </select>

          <label>Skin concern 2?</label>
          <select name="concern2" onChange={handleChange}>
            <option value="">Select</option>
            <option value="dryness">Dryness</option>
            <option value="sensitivity">Sensitivity</option>
            <option value="redness">Redness</option>
          </select>

          <label>Skin concern 3?</label>
          <select name="concern3" onChange={handleChange}>
            <option value="">Select</option>
            <option value="pore">Large Pores</option>
            <option value="texture">Uneven Texture</option>
            <option value="scarring">Scarring</option>
          </select>

          <label>Key Ingredient?</label>
          <select name="keyIngredient" onChange={handleChange}>
            <option value="">Select</option>
            <option value="vitaminC">Vitamin C</option>
            <option value="retinol">Retinol</option>
            <option value="hyaluronicAcid">Hyaluronic Acid</option>
          </select>

          <label>Formulation?</label>
          <select name="formulation" onChange={handleChange}>
            <option value="">Select</option>
            <option value="gel">Gel</option>
            <option value="cream">Cream</option>
            <option value="serum">Serum</option>
          </select>

          <Link to="/results"><button type="submit">Submit</button></Link>
        </form>
      </div>

      <div className="imager-x">
        <div><img className="image-6"src={image2} alt="" /></div>
        <div><img className="image-5"src={image1} alt="" /></div>
      </div>
    </div>
  );
};
export default Formpage;