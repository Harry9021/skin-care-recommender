import React, { useState } from "react";
import "../Styles/Formpage.css";
import image1 from "../Vectors/image5.png";
import image2 from "../Vectors/image6.png";
import { Link, useNavigate } from "react-router-dom";
import { IoMdArrowRoundBack } from "react-icons/io";

const Formpage = () => {
    const [formData, setFormData] = useState({
        skinType: "",
        concern1: "",
        concern2: "",
        concern3: "",
    });

    const navigate = useNavigate();

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value,
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        const userInput = {
            skin_type: getSkinTypeValue(formData.skinType),
            concern_1: getConcernValue(formData.concern1),
            concern_2: getConcernValue(formData.concern2),
            concern_3: getConcernValue(formData.concern3),
        };

        try {
            const response = await fetch(
                "http://localhost:4000/get-recommendations",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(userInput),
                }
            );

            const recommendations = await response.json();
            console.log("Recommended Products:", recommendations);
            // alert("Recommendations retrieved successfully");
            navigate("/results", { state: { recommendations } });

        } catch (error) {
            console.error("Error fetching recommendations:", error);
            alert("Failed to retrieve recommendations. Please try again.");
        }
    };

    const getSkinTypeValue = (skinType) => {
        console.log("calling...............");
        switch (skinType) {
            case "dry":
                return 0;
            case "oily":
                return 1;
            case "combination":
                return 2;
            case "normal":
                return 3;
            default:
                return 0;
        }
    };

    const getConcernValue = (concern) => {
        console.log("calling>>>>>>>>>>>");
        switch (concern) {
            case "anti-pollution":
                return 0;
            case "tan-removal":
                return 1;
            case "dryness":
                return 2;
            case "deep-nourishment":
                return 3;
            case "blackheads":
                return 4;
            case "oil-control":
                return 5;
            case "fine-lines":
                return 6;
            case "uneven-skin-tone":
                return 7;
            case "dark-spots":
                return 8;
            case "dark-circles":
                return 9;
            case "skin-tightening":
                return 10;
            case "under-eye":
                return 11;
            case "skin-inflammation":
                return 12;
            case "general-care":
                return 13;
            case "redness":
                return 14;
            case "skin-sagging":
                return 15;
            case "lightening":
                return 16;
            case "sun-protection":
                return 17;
            case "pigmentation":
                return 18;
            case "blackheads-removal":
                return 19;
            case "oily-skin":
                return 20;
            case "anti-ageing":
                return 21;
            case "hydration":
                return 22;
            case "dull-skin":
                return 23;
            case "uneven-texture":
                return 24;
            case "irregular-textures":
                return 25;
            case "pore-minimizing":
                return 26;
            case "excess-oil":
                return 27;
            case "daily-use":
                return 28;
            case "dullness":
                return 29;
            case "anti-acne-scarring":
                return 30;
            case "softening":
                return 31;
            case "acne":
                return 32;
            case "pore-care":
                return 33;
            default:
                return 0;
        }
    };

    const concerns = [
        "anti-pollution",
        "tan-removal",
        "dryness",
        "deep-nourishment",
        "blackheads",
        "oil-control",
        "fine-lines",
        "uneven-skin-tone",
        "dark-spots",
        "dark-circles",
        "skin-tightening",
        "under-eye",
        "skin-inflammation",
        "general-care",
        "redness",
        "skin-sagging",
        "lightening",
        "sun-protection",
        "pigmentation",
        "blackheads-removal",
        "oily-skin",
        "anti-ageing",
        "hydration",
        "dull-skin",
        "uneven-texture",
        "irregular-textures",
        "pore-minimizing",
        "excess-oil",
        "daily-use",
        "dullness",
        "anti-acne-scarring",
        "softening",
        "acne",
        "pore-care",
    ];

    const getAvailableConcerns = (selected) => {
        return concerns.filter((concern) => concern !== selected);
    };

    return (
        <div className="form-image-container">
            <div className="blue-box">
                <Link to="/">
                    <div className="backarrow">
                        <IoMdArrowRoundBack
                            style={{
                                width: 30,
                                height: 30,
                                marginLeft: 10,
                                marginTop: 10,
                            }}
                        />
                    </div>
                </Link>
            </div>
            <div className="form-container">
                <form onSubmit={handleSubmit}>
                    <h3>
                        Fill Below Information to know your skincare products.
                    </h3>

                    <label>What is your skin type?</label>
                    <select name="skinType" onChange={handleChange} required>
                        <option value="">Select</option>
                        <option value="dry">Dry</option>
                        <option value="oily">Oily</option>
                        <option value="combination">Combination</option>
                        <option value="normal">Normal</option>
                    </select>

                    <label>Skin concern 1?</label>
                    <select name="concern1" onChange={handleChange} required>
                        <option value="">Select</option>
                        {concerns.map((concern) => (
                            <option
                                key={concern}
                                value={concern}
                                disabled={
                                    formData.concern2 === concern ||
                                    formData.concern3 === concern
                                }
                            >
                                {concern
                                    .replace(/-/g, " ")
                                    .replace(/\b\w/g, (char) =>
                                        char.toUpperCase()
                                    )}
                            </option>
                        ))}
                    </select>

                    <label>Skin concern 2?</label>
                    <select name="concern2" onChange={handleChange} required>
                        <option value="">Select</option>
                        {getAvailableConcerns(formData.concern1).map(
                            (concern) => (
                                <option
                                    key={concern}
                                    value={concern}
                                    disabled={formData.concern3 === concern}
                                >
                                    {concern
                                        .replace(/-/g, " ")
                                        .replace(/\b\w/g, (char) =>
                                            char.toUpperCase()
                                        )}
                                </option>
                            )
                        )}
                    </select>

                    <label>Skin concern 3?</label>
                    <select name="concern3" onChange={handleChange} required>
                        <option value="">Select</option>
                        {getAvailableConcerns(formData.concern1)
                            .filter((concern) => concern !== formData.concern2)
                            .map((concern) => (
                                <option key={concern} value={concern}>
                                    {concern
                                        .replace(/-/g, " ")
                                        .replace(/\b\w/g, (char) =>
                                            char.toUpperCase()
                                        )}
                                </option>
                            ))}
                    </select>

                    <button type="submit">Submit</button>
                </form>
            </div>

            <div className="imager-x">
                <div>
                    <img className="image-6" src={image2} alt="" />
                </div>
                <div>
                    <img className="image-5" src={image1} alt="" />
                </div>
            </div>
        </div>
    );
};

export default Formpage;
