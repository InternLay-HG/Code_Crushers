package com.QueLet.QueLet.Controller;

import com.QueLet.QueLet.Model.Business;
import com.QueLet.QueLet.Model.CategoryType;
import com.QueLet.QueLet.Repository.BusinessRepository;
import com.QueLet.QueLet.Service.BusinessService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
@RestController
@RequestMapping("/business")
public class BusinessController {
    @Autowired
    private BusinessService businessService;
    @Autowired
    private BusinessRepository businessRepository;

    @GetMapping("/All-business")
    public List<Business> getAllBusiness(){
        return businessService.AllBusiness();
    }
    @GetMapping("/business-category")
    public ResponseEntity<List<Business>> getBusinessesByCategoryType(@RequestParam CategoryType categoryType) {
        List<Business> businesses = businessService.getBusinessType(categoryType);
        return ResponseEntity.ok(businesses);
    }

    @PutMapping(value = "/update-business/{id}", consumes = "application/json")
    public ResponseEntity<Business> updateProduct(@PathVariable Long id, @RequestBody Business updatedBusiness) {
        try {
            Business business = businessService.UpdateBusiness(id, updatedBusiness);
            return ResponseEntity.ok(business);
        } catch (RuntimeException e) {
            return ResponseEntity.status(404).build();
        }
    }
}
