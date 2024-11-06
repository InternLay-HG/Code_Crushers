package com.QueLet.QueLet.Service;

import com.QueLet.QueLet.Model.Business;
import com.QueLet.QueLet.Model.Customer;
import com.QueLet.QueLet.Repository.BusinessRepository;
import com.QueLet.QueLet.Repository.CustomerRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

@Service
public class CustomUserDetailsService implements UserDetailsService {

    @Autowired
    private CustomerRepository customerRepository;
    @Autowired
    private BusinessRepository businessRepository;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        // Load user from the database by username
//        Customer user = customerRepository.findByUsername(username)
//                .orElseThrow(() -> new UsernameNotFoundException("User not found with username: " + username));
//        return user;
  //  }

        Customer customer = customerRepository.findByUsername(username).orElse(null);
        if (customer != null) {
            return customer;
        }

        // If no Customer is found, attempt to load a Business by username
        Business business = businessRepository.findByUsername(username).orElse(null);
        if (business != null) {
            return business;
        }

        // Throw exception if no user found
        throw new UsernameNotFoundException("User not found with username: " + username);
    }
}