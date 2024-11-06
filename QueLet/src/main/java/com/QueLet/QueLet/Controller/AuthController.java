package com.QueLet.QueLet.Controller;

import com.QueLet.QueLet.Model.Business;
import com.QueLet.QueLet.Model.Customer;
import com.QueLet.QueLet.Model.JwtRequest;
import com.QueLet.QueLet.Model.JwtResponse;
import com.QueLet.QueLet.Security.JwtUtil;
import com.QueLet.QueLet.Service.BusinessService;
import com.QueLet.QueLet.Service.CustomUserDetailsService;
import com.QueLet.QueLet.Service.CustomerService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/auth")
@CrossOrigin(origins = "http://localhost:3000")
public class AuthController {

    @Autowired
    private AuthenticationManager authenticationManager;

    @Autowired
    private JwtUtil jwtUtil;

    @Autowired
    private CustomUserDetailsService userDetailsService;
    @Autowired
    private CustomerService userService;
    @Autowired
    private BusinessService businessService;

@PostMapping("/customer-signup")
public ResponseEntity<?> signup(@RequestBody Customer user) {
    try {
        Customer newUser = userService.save(user);
        return new ResponseEntity<>(newUser, HttpStatus.CREATED);

    } catch (IllegalArgumentException e) {
        // Handles cases such as "username already taken" or "password cannot be null or empty"
        return new ResponseEntity<>(e.getMessage(), HttpStatus.CONFLICT);

    } catch (Exception e) {
        // General error response for any other unexpected issues
        return new ResponseEntity<>("An error occurred during signup. Please try again later.", HttpStatus.BAD_REQUEST);
    }
}
    @PostMapping("/customer-login")
    public ResponseEntity<JwtResponse> login(@RequestBody JwtRequest request) throws Exception {
        try {
            authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(request.getUsername(), request.getPassword())
            );
        } catch (Exception e) {
            throw new Exception("Invalid credentials", e);
        }

        UserDetails userDetails = userDetailsService.loadUserByUsername(request.getUsername());
        String token = this.jwtUtil.generateToken(userDetails.getUsername());

        JwtResponse response = JwtResponse.builder()
                .jwtToken(token)
                .username(userDetails.getUsername()).build();
        return new ResponseEntity<>(response, HttpStatus.OK);
    }

    @PostMapping("/business-signup")
    public ResponseEntity<?> businessSignup(@RequestBody Business user) {
        try {
            Business newUser = businessService.save(user);
            return new ResponseEntity<>(newUser, HttpStatus.CREATED);

        } catch (IllegalArgumentException e) {
            // Handles cases such as "username already taken" or "password cannot be null or empty"
            return new ResponseEntity<>(e.getMessage(), HttpStatus.CONFLICT);

        } catch (Exception e) {
            // General error response for any other unexpected issues
            return new ResponseEntity<>("An error occurred during signup. Please try again later.", HttpStatus.BAD_REQUEST);
        }
    }

    @PostMapping("/business-login")
    public ResponseEntity<JwtResponse> businessLogin(@RequestBody JwtRequest request) throws Exception {
        try {
            authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(request.getUsername(), request.getPassword())
            );
        } catch (Exception e) {
            throw new Exception("Invalid credentials", e);
        }

        UserDetails userDetails = userDetailsService.loadUserByUsername(request.getUsername());
        String token = this.jwtUtil.generateToken(userDetails.getUsername());

        JwtResponse response = JwtResponse.builder()
                .jwtToken(token)
                .username(userDetails.getUsername()).build();
        return new ResponseEntity<>(response, HttpStatus.OK);
    }

    @GetMapping("/All-User")
    public List<Customer> getAllCustomer(){
        return userService.AllUser();
    }
}
